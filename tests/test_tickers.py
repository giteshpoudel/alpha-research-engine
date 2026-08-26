from src.ingestion.tickers import extract_tickers


def test_cashtags():
    assert extract_tickers("$BTC and $eth are pumping") == ["BTC", "ETH"]


def test_aliases_case_insensitive():
    assert extract_tickers("Bitcoin rallies while Ethereum dumps") == ["BTC", "ETH"]


def test_dedup_and_sorted():
    assert extract_tickers("bitcoin BTC $btc Bitcoin") == ["BTC"]


def test_no_false_positives_on_dollar_amounts():
    assert extract_tickers("Price target is $100 this year") == []


def test_alias_word_boundaries():
    # "ether" alone is not an alias; "solana" is, "solace" is not
    assert extract_tickers("The ether between us, what solace") == []
    assert extract_tickers("solana summer") == ["SOL"]


def test_no_ambiguous_aliases():
    # 'link', 'dot', 'ada' as plain words must NOT match (chainlink/polkadot/cardano only)
    assert extract_tickers("check this link about the dot com era, ada lovelace") == []
    assert extract_tickers("chainlink and polkadot and cardano") == ["ADA", "DOT", "LINK"]
