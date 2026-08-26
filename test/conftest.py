def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: downloads multi-GB HF checkpoints and runs full inference",
    )
    config.addinivalue_line(
        "markers",
        "gpu: needs CUDA and flash-attn (the paged attention path)",
    )


# Fakes shared by the engine and benchmark suites. A FakeRunner stands in
# for the GPU, so everything above the model runs on a laptop.
EOS = 0


class FakeTokenizer:
    """One token per character, so tokens and text line up by eye."""

    eos_token_id = EOS

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(t) for t in token_ids if t != EOS)


class FakeRunner:
    """Replays a fixed script of tokens, the same one for every sequence."""

    def __init__(self, tokens: list[int]):
        self.tokens = tokens
        self.step = 0

    def forward(self, batch):
        return None

    def sample(self, logits, batch):
        token = self.tokens[min(self.step, len(self.tokens) - 1)]
        self.step += 1
        return [token] * len(batch.seqs)
