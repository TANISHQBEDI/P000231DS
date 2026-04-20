
import pandas as pd
import re


class TextCleaner:
    DEFAULT_PHRASE_MAP = {
        'nff': 'no fault found',
        'u/s': 'unserviceable',
        'a/c': 'aircraft',
        'acft': 'aircraft',
        'flt': 'flight',
        'fwd': 'forward',
        'lh': 'left hand',
        'rh': 'right hand',
        'assy': 'assembly',
        'insp': 'inspection',
        'btw': 'between',
    }

    DEFAULT_REGEX_RULES = [
        (r'\bi\.a\.w\.\b', 'iaw'),
        (r'\br\s*[+&/]\s*r\b', 'removed and replaced'),
        (r'\bops?\s*chk\b', 'operational check'),
        (r'\bops?\s*check\b', 'operational check'),
        (r'\biaw\b', 'in accordance with'),
        (r'\bamm\b', 'aircraft maintenance manual'),
        (r'\bsrm\b', 'structural repair manual'),
        (r'\btsm\b', 'troubleshooting manual'),
        (r'\bw/o\b', 'work order'),
        (r'\bp/n\b', 'part number'),
        (r'\bs/n\b', 'serial number'),
        (r'\bc/w\b', 'complied with'),
        (r'\bstrg\b', 'stringer'),
        (r'\bstr\b', 'stringer'),
        (r'\bfr\b', 'frame'),
        (r'\bemergecny\b', 'emergency'),
        (r'\bemerensy\b', 'emergency'),
        (r'\bconector\b', 'connector'),
        (r'\bguset\b', 'gusset'),
        (r'\bstriner\b', 'stringer'),
        (r'\bdampning\b', 'damping'),
        (r'\brepaird\b', 'repaired'),
        (r'\bmecha\s*nical\b', 'mechanical'),
    ]

    def __init__(self, data: pd.DataFrame):
        self.__data = data
    
    def get_data(self) -> pd.DataFrame:
        return self.__data
    
    def remove_duplicates(self, columns: list[str] = ['discrepancy']) -> 'TextCleaner':
        self.__data = self.__data.drop_duplicates(subset=[*columns])
        return self

    def remove_null(self, columns: list[str] = ['discrepancy']) -> 'TextCleaner':
        self.__data = self.__data.dropna(subset=[*columns])
        return self

    @staticmethod
    def _basic_normalize(text: str) -> str:
        text = str(text).strip().lower()
        text = re.sub(r'\s+', ' ', text)
        return text

    def clean_discrepancy_text(
        self,
        column: str = 'discrepancy',
        output_column: str = 'discrepancy_clean',
        phrase_map: dict[str, str] | None = None,
        regex_rules: list[tuple[str, str]] | None = None,
    ) -> 'TextCleaner':
        """
        Clean free-text discrepancy values using deterministic rules.

        phrase_map:
            Exact phrase replacements after lowercase normalization.
            Example: {"nff": "no fault found"}

        regex_rules:
            Ordered regex replacements as (pattern, replacement).
            Example: [(r"\\bo/?h\\b", "overhaul")]

        If phrase_map/regex_rules are None, built-in defaults are used.
        """
        if column not in self.__data.columns:
            raise ValueError(f"Column '{column}' not found")

        cleaned = self.__data[column].fillna('').astype(str).map(self._basic_normalize)

        if phrase_map is None:
            phrase_map = self.DEFAULT_PHRASE_MAP

        if regex_rules is None:
            regex_rules = self.DEFAULT_REGEX_RULES

        if phrase_map:
            normalized_map = {
                self._basic_normalize(k): self._basic_normalize(v)
                for k, v in phrase_map.items()
            }
            cleaned = cleaned.replace(normalized_map)

        if regex_rules:
            for pattern, replacement in regex_rules:
                cleaned = cleaned.str.replace(
                    pattern,
                    replacement,
                    regex=True,
                )

        self.__data[output_column] = cleaned.str.replace(r'\s+', ' ', regex=True).str.strip()
        return self

    def pipe(
        self,
        source_column: str = 'discrepancy',
        cleaned_column: str = 'discrepancy_clean',
        phrase_map: dict[str, str] | None = None,
        regex_rules: list[tuple[str, str]] | None = None,
        use_default_rules: bool = True,
    ):
        self = self.remove_duplicates().remove_null()

        if use_default_rules or phrase_map is not None or regex_rules is not None:
            self = self.clean_discrepancy_text(
                column=source_column,
                output_column=cleaned_column,
                phrase_map=phrase_map,
                regex_rules=regex_rules,
            )
        return self.get_data()


if __name__ == '__main__':
    from src.utils.paths import RAW_FILE
    df = pd.read_csv(RAW_FILE)
    tp = TextCleaner(df)
    tp = tp.remove_duplicates().remove_null()