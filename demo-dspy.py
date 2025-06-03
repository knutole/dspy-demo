import dspy
import json

# configure the LLM
dspy.configure(lm=dspy.LM("gpt-4.1-mini-2025-04-14", max_tokens=4096, cache=False))


# define your initial prompt
class PIISignature(dspy.Signature):
    """
    Remove PII and inappropriate content from text.
    Also remove identifying information in terms of location, exact times, product name and type, store name, etc.
    Replace redacted text with generalized words or just omit the word if grammatically possible.
    Remove references to language problems, nationality, religion, gender, age, etc.
    Keep references to logistics companies like Porterbuddy, HeltHjem, Postnord, Posten, Bring, DHL, FedEx, etc.
    Output in same language as input.
    """

    # define input
    text: str = dspy.InputField(description="The text to clean")

    # define output
    clean_text: str = dspy.OutputField(description="The cleaned text")


# define your program, a so-called module
class PIIRemover(dspy.Module):
    def __init__(self):
        super().__init__()

        # using ChainOfThough module, with our signature. there are many other modules
        self.chain = dspy.ChainOfThought(PIISignature)

    def forward(self, text):

        # text, since that's our INPUT in our signature
        return self.chain(text=text)


# define your metric. this is just a function that compares
# the result to the expected result in the training dataset
def pii_metric(gold, pred, trace=None) -> float:
    """
    Metric function for evaluating PII removal performance.
    Returns 1.0 if all redactions are removed, 0.0 otherwise.
    """

    clean_text = getattr(pred, "clean_text", None)
    redactions = getattr(gold, "redactions", None)

    # check if any of the expected redactions are still in the text
    for redaction in redactions:
        if redaction.lower() in clean_text.lower():
            return 0.0

    return 1.0


# helper function to format training data
def create_trainset(path):
    with open(path, "r") as f:
        data = json.load(f)

    examples = []
    for item in data:
        examples.append(
            # text = INPUT in our signature, redactions we use in our metric method
            dspy.Example(text=item["text"], redactions=item["redactions"]).with_inputs(
                # define the text key as our input in this with_inputs method
                "text"
            )
        )

    # split the dataset into training, dev
    size = int(len(examples) * 0.8)
    return examples[:size], examples[size:]


def optimize_program():

    # just get the training data in the right format
    trainset, devset = create_trainset("training-data.json")

    # get your program
    program = PIIRemover()

    # define optimizer (this one is MIPROv2)
    optimizer = dspy.MIPROv2(metric=pii_metric, auto="light")

    # optimize your program!
    optimized_program = optimizer.compile(program.deepcopy(), trainset=trainset)

    # save your results
    optimized_program.save("optimized_program2.json")


def use_optimized_program():

    # load your optimized program
    optimized_program = PIIRemover()
    optimized_program.load("optimized_program.json")

    # use your program
    result = optimized_program("Jeg elsker Posten. Ring mæ! Hilsen Rune, tlf 48484848")

    print(result.clean_text)  # Jeg elsker Posten. Ring meg!


if __name__ == "__main__":

    # run your optimization
    optimize_program()

    # run your optimized program
    # use_optimized_program()
