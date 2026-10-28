"""Quick smoke tests that do not require the model to be running."""
from tools import calculator

def run_calculator(expression: str):
    # LangChain @tool objects expose invoke().
    return calculator.invoke({"expression": expression})

def test_calculator():
    assert run_calculator("2 + 2") == "4"
    assert run_calculator("sqrt(16)") == "4.0"
    assert run_calculator("12500 * 37") == "462500"
    assert "Unsupported expression" in run_calculator("__import__('os').system('echo bad')")

if __name__ == "__main__":
    test_calculator()
    print("Calculator smoke tests passed.")
    print("For full agent test, run: python test_agent.py")
