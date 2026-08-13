# Glossary

This document defines Yurine's core terminology.

| Term                      | Definition                                                                                                    |
| ------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Token**                 | A user-provided value treated as an indivisible unit of comparison and indexing.                              |
| **Sequence**              | A finite, ordered list of tokens.                                                                             |
| **Segment**               | A non-empty contiguous portion of a sequence, identified by a range of token positions.                       |
| **Vocabulary**            | The mapping between each distinct token in the data sequences and the symbol assigned to it.                  |
| **Symbol**                | A compact integer identifier used internally to represent a token; it is meaningful only within its encoding. |
| **String**                | The internal representation of a sequence as a finite, ordered list of symbols.                               |
| **Substring**             | A non-empty contiguous portion of a string, identified by a range of symbol positions.                        |
| **Alphabet**              | The set of distinct symbols occurring in the data strings of a corpus.                                        |
| **Data sequence/string**  | A sequence submitted for indexing, or the string produced by encoding it.                                     |
| **Query sequence/string** | A sequence submitted as a search query, or the string produced by encoding it for that search.                |
| **Corpus**                | The ordered collection of strings produced from indexed data sequences.                                       |
