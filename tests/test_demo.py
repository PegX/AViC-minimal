import json
from avic_gatn.demo import run

def test_demo_is_repeatable_and_serializable():
    a=run();b=run()
    assert a == b
    assert a["project"] == "AViC"
    json.dumps(a,allow_nan=False)
