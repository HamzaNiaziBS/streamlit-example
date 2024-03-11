import json
from langchain.document_loaders.base import BaseLoader
from typing import Any, Callable, Dict, List, Optional, Union
from langchain.schema.document import Document


class JSONLoaderAPI(BaseLoader):
        """Load a `JSON` file using a `jq` schema.

        Example:
            [{"text": ...}, {"text": ...}, {"text": ...}] -> schema = .[].text
            {"key": [{"text": ...}, {"text": ...}, {"text": ...}]} -> schema = .key[].text
            ["", "", ""] -> schema = .[]
        """

        def __init__(
            self,
            jq_schema: str,
            content_key: Optional[str] = None,
            metadata_func: Optional[Callable[[Dict, Dict], Dict]] = None,
            text_content: bool = True,
            json_lines: bool = False,
            json_object: str = "",
        ):
            """Initialize the JSONLoader.

            Args:
                jq_schema (str): The jq schema to use to extract the data or text from
                    the JSON.
                content_key (str): The key to use to extract the content from the JSON if
                    the jq_schema results to a list of objects (dict).
                metadata_func (Callable[Dict, Dict]): A function that takes in the JSON
                    object extracted by the jq_schema and the default metadata and returns
                    a dict of the updated metadata.
                text_content (bool): Boolean flag to indicate whether the content is in
                    string format, default to True.
                json_lines (bool): Boolean flag to indicate whether the input is in
                    JSON Lines format.
            """
            try:
                import jq  # noqa:F401
            except ImportError:
                raise ImportError(
                    "jq package not found, please install it with `pip install jq`"
                )

            self._jq_schema = jq.compile(jq_schema)
            self._content_key = content_key
            self._metadata_func = metadata_func
            self._text_content = text_content
            self._json_lines = json_lines
            self._json_object = json_object

        def load(self) -> List[Document]:
            """Load and return documents from the JSON file."""
            # print(self._json_object)
            docs: List[Document] = []

            self._parse(self._json_object, docs)
            return docs

        def _parse(self, content: str, docs: List[Document]) -> None:
            """Convert given content to documents."""
            data = self._jq_schema.input(json.loads(content))

            # Perform some validation
            # This is not a perfect validation, but it should catch most cases
            # and prevent the user from getting a cryptic error later on.
            if self._content_key is not None:
                self._validate_content_key(data)
            if self._metadata_func is not None:
                self._validate_metadata_func(data)

            for i, sample in enumerate(data, len(docs) + 1):
                text = self._get_text(sample=sample)
                metadata = self._get_metadata(
                    sample=sample, source="BS_CRM_Opportunities_Notes", seq_num=i
                )
                docs.append(Document(page_content=text, metadata=metadata))

        def _get_text(self, sample: Any) -> str:
            """Convert sample to string format"""
            if self._content_key is not None:
                content = sample.get(self._content_key)
            else:
                content = sample

            if self._text_content and not isinstance(content, str):
                raise ValueError(
                    f"Expected page_content is string, got {type(content)} instead. \
                        Set `text_content=False` if the desired input for \
                        `page_content` is not a string"
                )

            # In case the text is None, set it to an empty string
            elif isinstance(content, str):
                return content
            elif isinstance(content, dict):
                return json.dumps(content) if content else ""
            else:
                return str(content) if content is not None else ""

        def _get_metadata(
            self, sample: Dict[str, Any], **additional_fields: Any
        ) -> Dict[str, Any]:
            """
            Return a metadata dictionary base on the existence of metadata_func
            :param sample: single data payload
            :param additional_fields: key-word arguments to be added as metadata values
            :return:
            """
            if self._metadata_func is not None:
                return self._metadata_func(sample, additional_fields)
            else:
                return additional_fields

        def _validate_content_key(self, data: Any) -> None:
            """Check if a content key is valid"""
            sample = data.first()
            if not isinstance(sample, dict):
                raise ValueError(
                    f"Expected the jq schema to result in a list of objects (dict), \
                        so sample must be a dict but got `{type(sample)}`"
                )

            if sample.get(self._content_key) is None:
                raise ValueError(
                    f"Expected the jq schema to result in a list of objects (dict) \
                        with the key `{self._content_key}`"
                )

        def _validate_metadata_func(self, data: Any) -> None:
            """Check if the metadata_func output is valid"""

            sample = data.first()
            if self._metadata_func is not None:
                sample_metadata = self._metadata_func(sample, {})
                if not isinstance(sample_metadata, dict):
                    raise ValueError(
                        f"Expected the metadata_func to return a dict but got \
                            `{type(sample_metadata)}`"
                    )