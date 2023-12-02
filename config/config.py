import logging
import os

CHUNK_SIZE = 500
OVERLAP_SIZE = 50
ZH_TITLE_ENHANCE = False
log_verbose = False
LLM_MODELS = ["chatglm2"]

NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")

LOG_FORMAT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)


LOADER_DICT = {
    "UnstructuredHTMLLoader": [".html"],
    "UnstructuredMarkdownLoader": [".md"],
    "CustomJSONLoader": [".json"],
    "CSVLoader": [".csv"],
    # "FilteredCSVLoader": [".csv"],
    "RapidOCRPDFLoader": [".pdf"],
    "RapidOCRLoader": [".png", ".jpg", ".jpeg", ".bmp"],
    "UnstructuredFileLoader": [
        ".eml",
        ".msg",
        ".rst",
        ".rtf",
        ".txt",
        ".xml",
        ".docx",
        ".epub",
        ".odt",
        ".ppt",
        ".pptx",
        ".tsv",
    ],
}
SUPPORTED_EXTS = [ext for sublist in LOADER_DICT.values() for ext in sublist]

TEXT_SPLITTER_NAME = "ChineseRecursiveTextSplitter"

text_splitter_dict = {
    "ChineseRecursiveTextSplitter": {
        "source": "huggingface",  ## 选择tiktoken则使用openai的方法
        "tokenizer_name_or_path": "",
    },
    "SpacyTextSplitter": {
        "source": "huggingface",
        "tokenizer_name_or_path": "gpt2",
    },
    "RecursiveCharacterTextSplitter": {
        "source": "tiktoken",
        "tokenizer_name_or_path": "cl100k_base",
    },
    "MarkdownHeaderTextSplitter": {
        "headers_to_split_on": [
            ("#", "head1"),
            ("##", "head2"),
            ("###", "head3"),
            ("####", "head4"),
        ]
    },
}
