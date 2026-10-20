from mol_lpgm.schedule import scheduled_learning_rate


def rate(step):
    return scheduled_learning_rate(
        step,
        schedule="exponential-warmdown",
        peak_learning_rate=1e-4,
        minimum_learning_rate=1e-9,
        start_step=0,
        end_step=400_000,
        warmup_steps=2_180,
        warmdown_steps=50_000,
        exponential_decay_factor=0.175670801936739,
    )


def test_production_schedule_boundaries():
    assert rate(0) == 0
    assert rate(2_180) == 1e-4
    assert abs(rate(350_000) - 1.75670801936739e-5) < 1e-15
    assert rate(400_000) == 1e-9
    assert rate(349_999) > rate(350_000) > rate(350_001)
