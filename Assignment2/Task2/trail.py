# read .npz file 
import numpy as np
@dataclass
class KeywordCandidate:
    field_id: int
    start_idx: int
    end_idx: int
    field_type: FieldType
    
    def __len__(self):
        return self.end_idx - self.start_idx + 1


def load_alignment_and_candidates_npz(filepath):
    data = np.load(filepath, allow_pickle=True)
    return data["aligned"].tolist(), data["candidates"].tolist()


msgs_aligned, key_word_candidates = load_alignment_and_candidates_npz("client_alignment_and_candidates.npz")

print(key_word_candidates[:3])


print(keyword_candidates[:5])
