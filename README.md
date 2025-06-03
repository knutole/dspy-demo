# DSPy demo

## Install

```bash
# create virtual environment for python
uv venv .venv
source .venv/bin/activate

# install dependencies
uv pip install -r requirements.txt

# add .env
cp .env.example .env

# add key
echo "OPENAI_API_KEY=your_api_key" > .env
```

## Run

```bash
uv run python demo-dspy.py
```

## Learn

### Create training dataset

#### Make your own dataset

What you need is 1) examples to try your program on, and 2) some metric for what is the correct way to process your example. Look at `training-data.json` for example. There we define two things: 1) "text" key with an unwashed comment, and 2) a "redactions" key with a list of words that we want redacted from the text. Writing a metric method to check our work, is then simply a matter of comparing the washed text with the desired redactions.

#### Convert your dataset to DSPy-compatible format

Define a method to convert your JSON (or whatever) dataset to `dspy.Example` format:

```py

def create_trainset(path):
    with open(path, "r") as f:
        data = json.load(f)

    examples = []
    for item in data:
        examples.append(

            # create an instance of dspy.Example
            dspy.Example(text=item["text"], redactions=item["redactions"]).with_inputs(
                "text"
            )
        )

    size = int(len(examples) * 0.8)
    return examples[:size], examples[size:]
```

### Create your metric function

This function is run for every single example that is processed, to check if it was processed perfectly or not. You can define this function in whatever way you want, even calling another LLM to determine the quality of the processed example.

In our dataset, we have already defined clear criteria for what we want, so the metric function is as simple as checking for substrings in our text.

```py
def pii_metric(gold, pred, trace=None) -> float:
    """
    Metric function for evaluating PII removal performance.
    Returns 1.0 if all redactions are removed, 0.0 otherwise.

    Some modules require True/False metric, while MIPROv2 happens to require a float.
    """

    clean_text = getattr(pred, "clean_text", None)
    redactions = getattr(gold, "redactions", None)

    # check if any of the expected redactions are still in the text
    for redaction in redactions:
        if redaction.lower() in clean_text.lower():
            return 0.0

    return 1.0
```

### Optimized program

Look at `optimized_program.json`. This is the new, optimized prompt and examples, which you can use later.

#### Using an optimized program

You actually need your Signature and Module clas definitions to be present in your file, even if you did optimization somewhere else. This is because you are using the Module program (in this case `PIIRemover`) to load your saved, optimized program description.

```py
class PIISignature(dspy.Signature):
    """
    Remove PII and inappropriate content from text.
    Also remove identifying information in terms of location, exact times, product name and type, store name, etc.
    Replace redacted text with generalized words or just omit the word if grammatically possible.
    Remove references to language problems, nationality, religion, gender, age, etc.
    Keep references to logistics companies like Porterbuddy, HeltHjem, Postnord, Posten, Bring, DHL, FedEx, etc.
    Output in same language as input.
    """

    text: str = dspy.InputField(description="The text to clean")
    clean_text: str = dspy.OutputField(description="The cleaned text")


class PIIRemover(dspy.Module):
    def __init__(self):
        super().__init__()
        self.chain = dspy.ChainOfThought(PIISignature)

    def forward(self, text):
        return self.chain(text=text)


# load your optimized program
optimized_program = PIIRemover()
optimized_program.load("optimized_program.json")

# use your program
result = optimized_program("Jeg elsker Posten. Ring mæ! Hilsen Rune, tlf 48484848")

print(result.clean_text)  # Jeg elsker Posten. Ring meg!


```
