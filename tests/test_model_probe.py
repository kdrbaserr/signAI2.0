"""Offline checks for target discovery, PEFT integration and weight policy."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="Install the optional probe extra for model compatibility tests")
peft = pytest.importorskip("peft", reason="Install the optional probe extra for model compatibility tests")
from torch import nn

from signai.smoke_train import (
    ModelProbeError, MomentClassifier, assert_trainable_policy, parameter_hashes,
    select_lora_targets,
)


class TinyMoment(nn.Module):
    """A CPU unit fixture, not a substitute for the real checkpoint probe."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(task_name="classification")
        attention = nn.Module()
        attention.q = nn.Linear(12, 12, bias=False)
        attention.k = nn.Linear(12, 12, bias=False)
        attention.v = nn.Linear(12, 12, bias=False)
        layer = nn.Module()
        layer.SelfAttention = attention
        block = nn.Module()
        block.layer = nn.ModuleList([layer])
        self.encoder = nn.Module()
        self.encoder.block = nn.ModuleList([block])
        self.head = nn.Sequential(nn.Linear(12, 5))

    def forward(self, *, x_enc, input_mask=None):
        attention = self.encoder.block[0].layer[0].SelfAttention
        return SimpleNamespace(logits=self.head(attention.q(x_enc) + attention.v(x_enc)))


def test_targets_are_exact_encoder_attention_linear_names():
    model = MomentClassifier(TinyMoment())
    targets = select_lora_targets(model, ["q", "v"])
    assert targets == [
        "moment.encoder.block.0.layer.0.SelfAttention.q",
        "moment.encoder.block.0.layer.0.SelfAttention.v",
    ]
    assert all(isinstance(dict(model.named_modules())[name], nn.Linear) for name in targets)


@pytest.mark.parametrize("suffixes", [["NOT_REAL"], ["q", "NOT_REAL"], [], ["q", "q"]])
def test_invalid_or_partial_target_selection_fails_before_optimizer(suffixes):
    with pytest.raises(ModelProbeError):
        select_lora_targets(MomentClassifier(TinyMoment()), suffixes)


def test_adapter_and_head_update_while_base_is_unchanged():
    torch.manual_seed(42)
    moment = TinyMoment()
    moment.requires_grad_(False)
    wrapper = MomentClassifier(moment)
    targets = select_lora_targets(wrapper, ["q", "v"])
    model = peft.get_peft_model(wrapper, peft.LoraConfig(
        r=2, lora_alpha=4, bias="none", target_modules=targets, modules_to_save=["moment.head"],
    ))
    groups = assert_trainable_policy(model)
    parameters = dict(model.named_parameters())
    before = parameter_hashes(parameters.items())
    optimizer = torch.optim.AdamW([p for p in parameters.values() if p.requires_grad], lr=0.01, weight_decay=0)
    logits = model(x_enc=torch.randn(2, 12)).logits
    loss = nn.BCEWithLogitsLoss()(logits, torch.ones(2, 5))
    loss.backward()
    assert all(parameters[name].grad is None for name in groups["frozen"])
    assert any(parameters[name].grad.norm() > 0 for name in groups["lora"])
    assert all(parameters[name].grad is not None for name in groups["head"])
    optimizer.step()
    after = parameter_hashes(parameters.items())
    assert all(before[name] == after[name] for name in groups["frozen"])
    assert any(before[name] != after[name] for name in groups["lora"])
    assert any(before[name] != after[name] for name in groups["head"])
    assert isinstance(moment.config, SimpleNamespace)  # PEFT wrapper did not replace MOMENT's config.


def test_unexpected_trainable_base_parameter_is_rejected():
    with pytest.raises(ModelProbeError, match="Unexpected trainable base"):
        assert_trainable_policy(MomentClassifier(TinyMoment()))
