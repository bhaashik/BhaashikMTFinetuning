"""NLLB language-token API compatibility tests without model downloads."""
import pytest

from bhaashik.modeling import add_language_token, require_language


class LegacyNllbTokenizer:
    def __init__(self):
        self.additional_special_tokens = ["bho_Deva", "hin_Deva", "mag_Deva"]
        self.unk_token_id = 3
        self.vocab = {"<unk>": 3, "<pad>": 1, "bho_Deva": 20,
                      "hin_Deva": 21, "mag_Deva": 22, "foo_Deva": 23}

    @property
    def all_special_tokens(self):
        return ["<unk>", "<pad>"] + self.additional_special_tokens

    def get_vocab(self):
        return self.vocab

    def convert_tokens_to_ids(self, language):
        return self.vocab.get(language, self.unk_token_id)

    def add_special_tokens(self, mapping, replace_additional_special_tokens=True):
        assert replace_additional_special_tokens is False
        assert list(mapping) == ["additional_special_tokens"]
        for token in mapping["additional_special_tokens"]:
            self.additional_special_tokens.append(token)
            self.vocab[token] = max(self.vocab.values()) + 1
        return len(mapping["additional_special_tokens"])


class ModernNllbTokenizer(LegacyNllbTokenizer):
    def __init__(self):
        super().__init__()
        self.extra_special_tokens = self.additional_special_tokens
        del self.additional_special_tokens

    @property
    def all_special_tokens(self):
        return ["<unk>", "<pad>"] + self.extra_special_tokens

    def add_special_tokens(self, mapping, replace_extra_special_tokens=True):
        assert replace_extra_special_tokens is False
        assert list(mapping) == ["extra_special_tokens"]
        for token in mapping["extra_special_tokens"]:
            self.extra_special_tokens.append(token)
            self.vocab[token] = max(self.vocab.values()) + 1
        return len(mapping["extra_special_tokens"])


@pytest.mark.parametrize("tokenizer_class", [LegacyNllbTokenizer, ModernNllbTokenizer])
def test_registered_nllb_language_tokens(tokenizer_class):
    tokenizer = tokenizer_class()
    assert require_language(tokenizer, "bho_Deva") == 20
    assert require_language(tokenizer, "hin_Deva") == 21
    assert require_language(tokenizer, "mag_Deva") == 22


@pytest.mark.parametrize("tokenizer_class", [LegacyNllbTokenizer, ModernNllbTokenizer])
def test_rejects_vocab_token_that_is_not_registered_as_language(tokenizer_class):
    with pytest.raises(ValueError, match="Unsupported language token"):
        require_language(tokenizer_class(), "foo_Deva")


@pytest.mark.parametrize("tokenizer_class", [LegacyNllbTokenizer, ModernNllbTokenizer])
def test_rejects_missing_and_non_language_tokens(tokenizer_class):
    tokenizer = tokenizer_class()
    with pytest.raises(ValueError, match="Unsupported language token"):
        require_language(tokenizer, "xyz_Deva")
    with pytest.raises(ValueError, match="Invalid NLLB language tag"):
        require_language(tokenizer, "<unk>")


@pytest.mark.parametrize("tokenizer_class", [LegacyNllbTokenizer, ModernNllbTokenizer])
def test_extension_preserves_existing_language_tokens(tokenizer_class):
    tokenizer = tokenizer_class()
    assert add_language_token(tokenizer, "raj_Deva") == 1
    assert require_language(tokenizer, "raj_Deva") > 22
    assert require_language(tokenizer, "bho_Deva") == 20


def test_special_token_not_in_vocabulary_is_rejected():
    tokenizer = ModernNllbTokenizer()
    tokenizer.extra_special_tokens.append("xyz_Deva")
    with pytest.raises(ValueError, match="Unknown language token"):
        require_language(tokenizer, "xyz_Deva")
