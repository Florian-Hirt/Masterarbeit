# ogbg-code2

### attridx2attr.csv.gz

Mapping from attribute index to actual attribute term.

### typeidx2type.csv.gz

Mapping from type index to actual type term.

### graphidx2code.json.gz

Mapping from graph idx to raw code snippet.

```python
import pandas as pd 
code_dict = pd.read_json("graphidx2code.json.gz", orient='split')
## i-th code snippet can be accessed by
code_dict.iloc[i]
```