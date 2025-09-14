# CheckList Plus

**An LLM-enhanced extension of the original CheckList framework for behavioral testing of NLP models.**

This project extends the original [CheckList](https://github.com/marcotcr/checklist) framework with smarter and modern LLM capabilities, making it easier to create and run behavioral tests for NLP models.

## 🆕 What's New in CheckList Plus

### 🤖 LLM-Powered Text Generation & Perturbations

- **LLM Text Generator**: Generate context-aware text completions, paraphrases, and negations using OpenAI models.
- **Entity Detection & Masking**: Automatically detect and mask entities like colors, brands, and dates with spaCy and LLM integration.
- **Precision Perturbations**: Target specific numerical entities (e.g., `MONEY`, `DATE`, `QUANTITY`) for controlled text transformations.

### 🛠 Developer Experience Improvements

- **Unified API**: Consistent interface across all LLM-powered features
- **Rich Configuration**: YAML-based prompt configuration with template variable support
- **Comprehensive Examples**: Built-in examples for entity detection and other LLM tasks
- **Temperature Control**: Deterministic vs creative outputs with configurable temperature settings
- **Error Handling**: Graceful degradation and comprehensive error messaging

### 🔄 Backward Compatibility

- **100% Compatible**: All original CheckList functionality preserved and enhanced
- **Seamless Integration**: New LLM features integrate naturally with existing workflows
- **Optional Dependencies**: LLM features are optional - core functionality works without API keys

## 📖 Original Research

Based on the research paper:

> [Beyond Accuracy: Behavioral Testing of NLP models with CheckList](http://homes.cs.washington.edu/~marcotcr/acl20_checklist.pdf)
> Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, Sameer Singh
> *Association for Computational Linguistics (ACL), 2020*

```bibtex
@inproceedings{checklist:acl20,
  author = {Marco Tulio Ribeiro and Tongshuang Wu and Carlos Guestrin and Sameer Singh},
  title = {Beyond Accuracy: Behavioral Testing of NLP models with CheckList},
  booktitle = {Association for Computational Linguistics (ACL)},
  year = {2020}
}
```

### Advanced Use Cases

CheckList Plus extends behavioral testing beyond traditional NLP models to modern architectures:

- **[Testing Embeddings Behavior](notebooks/embeddings/Testing_Embeddings_Behaviour.ipynb)** - Evaluate embedding models by testing their ability to distinguish between paraphrases (should be similar) and negations (should be different). This notebook demonstrates how LLM-generated perturbations can reveal behavioral inconsistencies in embedding models.

*Inspired by research on embedding evaluation methodologies: ["Enhancing Negation Awareness in Universal Text Embeddings: A Data-efficient and Computational-efficient Approach"](https://arxiv.org/html/2504.00584v1)*

## 🚀 Quick Start

### Installation

```bash
pip install checklist-plus
```

### Key Features

#### 1. **LLM-Powered Text Generation**

```python
from checklist_plus.text_generation.llm import LLMTextGenerator

tg = LLMTextGenerator(openai_api_key="your-api-key", model_name="gpt-4o-mini")

# Smart paraphrasing with style control
paraphrases = tg.paraphrase(
    "The weather is nice today",
    n_paraphrases=3,
    style="formal",
    length_preference="longer",
)
# → ["Today's meteorological conditions are quite favorable",
#    "The atmospheric conditions are particularly pleasant today", ...]

# Intelligent negation
negated = tg.negate_sentence("I love this movie", n_variations=2)
# → ["I hate this movie", "I don't love this movie"]
```

#### 2. **Entity Detection & Masking**

```python
# Detect and mask entities
result = tg.detect_and_mask_entities(
    "I bought an iPhone for $999 yesterday", entity_type="brand names"
)
# → {
#     "original_text": "I bought an iPhone for $999 yesterday",
#     "masked_text": "I bought a [MASK] for $999 yesterday",
#     "contains_entities": True,
#     "entities": ["iPhone"]
# }
```

#### 3. **Precision Perturbations**

```python
from checklist_plus.perturb import Perturb
import spacy

nlp = spacy.load("en_core_web_sm")
data = ["The meeting is at 10:30 on Sept 14, tickets cost $45"]
parsed_data = list(nlp.pipe(data))

# Target specific entity types for number changes
ret = Perturb.perturb(
    parsed_data,
    Perturb.change_number,
    entity_types=["DATE", "MONEY"],  # Only change dates and money
    skip_abbreviations=False,  # Include numbers like '2' and '4'
    n=3,
)
# → Changes "14" to "16", "$45" to "$54", but preserves "10:30"
```

### Editor with LLM Integration

```python
# Initialize editor with LLM capabilities
editor = Editor()

# Traditional template generation (original feature)
templates = editor.template(
    "{first_name} is {a:profession} from {country}.",
    profession=["lawyer", "doctor", "accountant"],
)

# NEW: LLM-enhanced features through text generator
editor.tg = tg  # Attach LLM text generator

# Entity detection through editor
entities = editor.tg.detect_entities("Apple released the new MacBook", "brand names")
# → {"text": "Apple released the new MacBook", "contains_entities": True, "entities": ["Apple", "MacBook"]}
```

### Key Innovations Summary

**🎯 Precision Perturbations**: Instead of changing all numbers, target specific entity types (`MONEY`, `DATE`, `QUANTITY`) with spaCy NER integration.

**🤖 Structured LLM Outputs**: All LLM responses use Pydantic models for type safety and consistent data structures.

**🔄 Intelligent Fallbacks**: LLM methods automatically fall back to rule-based approaches for reliability.

**📝 Flexible Examples**: New `TextExample` class supports structured examples with input/output/description for better prompt engineering.

**🎨 Style-Aware Generation**: Paraphrasing and text generation with style control (`formal`, `casual`, `academic`, `business`).

**🔍 Entity Detection**: LLM-powered entity detection with configurable entity types and automatic masking.

**⚙️ Temperature Control**: Deterministic outputs (temperature=0) for entity detection, creative outputs for paraphrasing.

## Installation

From pypi:

```bash
pip install checklist-plus
jupyter nbextension install --py --sys-prefix checklist_plus.viewer
jupyter nbextension enable --py --sys-prefix checklist_plus.viewer
```

Note:  `--sys-prefix` to install into python’s sys.prefix, which is useful for instance in virtual environments, such as with conda or virtualenv. If you are not in such environments, please switch to `--user` to install into the user’s home jupyter directories.

From source:

```bash
git clone git@github.com:cowana-ai/checklist-plus.git
cd checklist-plus
pip install -e .
```

Either way, you need to install `pytorch` or `tensorflow` if you want to use masked language model suggestions:

```bash
pip install torch
```

For most tutorials, you also need to download a spacy model:

```bash
python -m spacy download en_core_web_sm
```

## 📚 Documentation

### Tutorials

1. [Generating data](notebooks/tutorials/1.%20Generating%20data.ipynb)
2. [Perturbing data (with LLM enhancements)](notebooks/tutorials/2.%20Perturbing%20data.ipynb)
3. [Test types and expectation functions](notebooks/tutorials/3.%20Test%20types,%20expectation%20functions,%20running%20tests.ipynb)
4. [The CheckList Plus process](notebooks/tutorials/4.%20The%20CheckList%20process.ipynb)

### Examples from Original Paper

- [Sentiment Analysis](notebooks/Sentiment.ipynb)
- [QQP (Question Pair Classification)](notebooks/QQP.ipynb)
- [SQuAD (Reading Comprehension)](notebooks/SQuAD.ipynb)

## 🔧 Advanced Installation

### From PyPI (Recommended)

```bash
pip install checklist-plus

# For Jupyter visualizations
jupyter nbextension install --py --sys-prefix checklist_plus.viewer
jupyter nbextension enable --py --sys-prefix checklist_plus.viewer
```

### From Source

```bash
git clone git@github.com:cowana-ai/checklist-plus.git
cd checklist-plus
pip install -e .
```

### Optional Dependencies

```bash
# For masked language model suggestions
pip install torch

# For NLP processing
python -m spacy download en_core_web_sm
```

## 💡 Key Features

### LLM-Enhanced Perturbations

```python
from checklist_plus.perturb import LLMPerturb

perturb = LLMPerturb(openai_api_key="your-key")

# Advanced negation with context
negated = perturb.add_negation_llm(
    ["I love programming", "This is excellent"], n_variations=2, context="casual"
)
```

### Enhanced Text Generation with LLM

```python
from checklist_plus.editor import Editor

# Initialize editor with LLM capabilities
llm_editor = Editor(
    use_llm=True, model_name="gpt-4o-mini", openai_api_key="your-api-key"
)

# Smart template filling with context
templates = llm_editor.template(
    "The {mask} is very {adj}.",
    adj=["beautiful", "interesting", "amazing"],
    context="travel destinations",
    n_completions=3,
)

# LLM-powered paraphrasing
paraphrases = llm_editor.paraphrase_llm(
    "The weather is beautiful today",
    n_paraphrases=3,
    style="formal",
    length_preference="longer",
)

# Context-aware word suggestions
suggestions = llm_editor.suggest("This is a {mask} movie.", context="science fiction")

# Smart synonyms and antonyms
synonyms = llm_editor.synonyms("The food is hot.", "hot")
antonyms = llm_editor.antonyms("The weather is cold.", "cold")
```

### Template Generation (Original Feature)

```python
from checklist_plus.editor import Editor

editor = Editor()
ret = editor.template(
    "{first_name} is {a:profession} from {country}.",
    profession=["lawyer", "doctor", "accountant"],
)
# → ['Mary is a doctor from Afghanistan.', 'Jordan is an accountant from Indonesia.', ...]
```

### Smart Perturbations

```python
from checklist_plus.perturb import Perturb
import spacy

nlp = spacy.load("en_core_web_sm")
data = ["John is a doctor", "Mary is a nurse"]
parsed_data = list(nlp.pipe(data))

# Rule-based perturbations (original)
ret = Perturb.perturb(parsed_data, Perturb.change_names, n=2)

# LLM-enhanced negation
ret_llm = perturb.add_negation_llm(["The service was good", "I liked the food"])
print(ret_llm)
```

### Test Creation and Execution

```python
from checklist_plus.test_types import MFT, INV, DIR
from checklist_plus.expect import Expect

# Minimum Functionality Tests
test1 = MFT(
    editor.template("This is {a:adj} {mask}.", adj=["good", "great"]).data,
    labels=1,
    name="Positive sentiment",
)

# Invariance Tests
test2 = INV(**Perturb.perturb(data, Perturb.add_typos))

# Directional Expectation Tests
test3 = DIR(
    **Perturb.perturb(data, add_negative_phrase),
    expect=Expect.monotonic(label=1, increasing=False)
)

# Run tests
test1.run(wrapped_model)
test1.summary()
```

## 🔗 Resources

- **[API Reference](https://checklist-nlp.readthedocs.io/en/latest/)** - Complete API documentation
- **[Original CheckList](https://github.com/marcotcr/checklist)** - The foundational framework
- **[Research Paper](http://homes.cs.washington.edu/~marcotcr/acl20_checklist.pdf)** - Original ACL 2020 paper
- **[Tutorial Notebooks](notebooks/tutorials/)** - Step-by-step guides

## 🤝 Contributing

This project extends the original CheckList framework. We welcome contributions that enhance LLM integration and improve usability while maintaining backward compatibility.

## 📄 License

This project follows the same license as the original CheckList framework.

______________________________________________________________________

**Note**: This is an extended version of the original [CheckList](https://github.com/marcotcr/checklist) framework with added LLM capabilities. All original functionality is preserved and enhanced.
