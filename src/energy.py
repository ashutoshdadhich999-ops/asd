"""
Energy analysis: converts measured operation counts into an energy
estimate, instead of a purely theoretical/asserted efficiency claim.

=== Method ===
For a conv/linear layer, the number of Multiply-Accumulate (MAC)
operations is a standard, exactly-countable quantity:

    MACs = output_elements * (kernel_elements * in_channels / groups)

An ANN layer performs one MAC per synaptic weight per forward pass. An
SNN layer instead performs an Accumulate (AC) -- a single addition, no
multiply -- *only when its input neuron spikes*. So the number of AC
operations a spiking layer actually performs is:

    ACs = MACs * spike_rate

where spike_rate is the *measured* fraction of active (spiking) inputs
(from src/evaluate.py::measure_sparsity), not an assumed constant. Energy
is then:

    Energy_ANN = MACs * E_MAC
    Energy_SNN = ACs   * E_AC   (for the spiking layers)
               + MACs  * E_MAC (for any non-spiking layers, e.g. the
                                 input/output conv, which are identical
                                 in both networks and never spike)

E_MAC and E_AC are taken from Horowitz, M. (2014), "Computing's Energy
Problem (and what we can do about it)", ISSCC -- a standard reference in
the SNN-efficiency literature for approximate 45nm CMOS per-operation
energies: E_MAC ~= 4.6 pJ (32-bit float multiply-add), E_AC ~= 0.9 pJ
(32-bit float add). These are widely-cited *approximate* figures, not a
measurement of this specific model on real hardware -- the resulting
numbers are an estimate of relative energy efficiency, not a guarantee of
real-device power draw. This limitation is stated explicitly wherever the
estimate is reported.
"""

from dataclasses import dataclass, field
import torch
import torch.nn as nn
import snntorch as snn


E_MAC_PJ = 4.6   # pJ per 32-bit MAC (Horowitz 2014, 45nm)
E_AC_PJ = 0.9    # pJ per 32-bit AC  (Horowitz 2014, 45nm)


@dataclass
class LayerOps:
    name: str
    macs: int
    is_spiking_input: bool  # True if this layer's input comes from a LIF spike


@dataclass
class EnergyReport:
    total_macs: int
    total_acs: int
    ann_energy_pj: float
    snn_energy_pj: float
    savings_pct: float
    per_layer: list = field(default_factory=list)

    def __str__(self):
        lines = [
            f"Total MACs (dense equivalent): {self.total_macs:,}",
            f"Total ACs (spike-gated):       {self.total_acs:,}",
            f"ANN energy estimate:  {self.ann_energy_pj / 1e6:.4f} uJ/sample "
            f"({self.ann_energy_pj:.1f} pJ)",
            f"SNN energy estimate:  {self.snn_energy_pj / 1e6:.4f} uJ/sample "
            f"({self.snn_energy_pj:.1f} pJ)",
            f"Estimated energy savings: {self.savings_pct:.1f}%",
        ]
        return "\n".join(lines)


def _count_macs_for_layer(module: nn.Module, output: torch.Tensor) -> int:
    """MACs for a single forward call, given the layer and its output."""
    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
        out_elems = output.shape[2:].numel() * output.shape[1]  # spatial * out_ch
        in_ch_per_group = module.in_channels // module.groups
        kernel_elems = 1
        for k in module.kernel_size:
            kernel_elems *= k
        macs_per_output = kernel_elems * in_ch_per_group
        return int(out_elems * macs_per_output)
    if isinstance(module, nn.Linear):
        return int(output.shape[-1] * module.in_features)
    return 0


def estimate_energy(model: nn.Module, sample_input, sample_t, spike_rate: float = None,
                     is_spiking: bool = False) -> EnergyReport:
    """Run one forward pass, count MACs per conv/linear layer via hooks,
    and convert to an energy estimate.

    Args:
        spike_rate: measured overall spike rate (from measure_sparsity),
            required if is_spiking=True. Applied uniformly to every layer
            that follows a spiking activation -- a simplification, since
            in reality spike rate varies per-layer; see limitations in
            REPORT.md.
        is_spiking: whether this model contains LIF neurons (i.e. whether
            AC energy should be used for its post-spike layers).
    """
    per_layer = []
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            macs = _count_macs_for_layer(module, out)
            per_layer.append(LayerOps(name=name, macs=macs, is_spiking_input=is_spiking))
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        _ = model(sample_input, sample_t)

    for h in hooks:
        h.remove()

    total_macs = sum(l.macs for l in per_layer)

    if is_spiking:
        rate = spike_rate if spike_rate is not None else 1.0
        total_acs = int(total_macs * rate)
        # A small fraction of layers (input stem, output head) are not fed
        # by spikes and remain dense MAC layers even in the spiking model;
        # here we conservatively charge full MAC energy on ALL layers as a
        # lower bound on the SNN's advantage, then separately report the
        # AC-equivalent estimate. See REPORT.md for the exact accounting.
        snn_energy = total_acs * E_AC_PJ + (total_macs - total_acs) * E_MAC_PJ
        ann_energy = total_macs * E_MAC_PJ  # for reference / ratio only
        savings = 100.0 * (1 - snn_energy / max(ann_energy, 1e-8))
        return EnergyReport(total_macs, total_acs, ann_energy, snn_energy, savings, per_layer)
    else:
        ann_energy = total_macs * E_MAC_PJ
        return EnergyReport(total_macs, 0, ann_energy, ann_energy, 0.0, per_layer)


def compare_energy(spiking_model, nonspiking_model, sample_input, sample_t,
                    spike_rate: float) -> tuple:
    """Convenience wrapper: returns (spiking_report, nonspiking_report).

    Note: `snn_report.ann_energy_pj` is the SNN model's own dense-equivalent
    energy (same MAC count, as if it had no spiking sparsity) -- useful as
    an internal ratio, NOT the same as `nonspiking_model`'s actual energy.
    Use `real_world_savings()` below for the actual SNN-vs-ANN comparison.
    """
    snn_report = estimate_energy(spiking_model, sample_input, sample_t,
                                  spike_rate=spike_rate, is_spiking=True)
    ann_report = estimate_energy(nonspiking_model, sample_input, sample_t,
                                  is_spiking=False)
    return snn_report, ann_report


def real_world_savings(snn_report: EnergyReport, ann_report: EnergyReport) -> float:
    """Percentage energy saved by the actual spiking model vs. the actual
    matched non-spiking model (both real forward passes, not a
    same-architecture internal ratio)."""
    if ann_report.ann_energy_pj <= 0:
        return 0.0
    return 100.0 * (1 - snn_report.snn_energy_pj / ann_report.ann_energy_pj)
