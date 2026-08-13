# Glossary

| 用語                      | 定義                                                                                     |
| ------------------------- | ---------------------------------------------------------------------------------------- |
| **token**                 | sequence を構成する値。Yurine における比較・索引化の最小単位。                           |
| **sequence**              | token の有限列。                                                                         |
| **segment**               | 1つの sequence 内の連続した非空の token range。                                          |
| **vocabulary**            | corpus へ追加された sequence に現れる distinct token value と vocabulary symbol の対応表 |
| **symbol**                | token を表す Yurine 内部の整数 ID                                                        |
| **string**                | symbol の有限列。sequence を symbol へ符号化した Yurine 内部の表現                       |
| **substring**             | 1つの string 内の連続した非空の symbol range                                             |
| **alphabet**              | vocabulary に含まれる symbol の集合                                                      |
| **data sequence/string**  | 索引に登録される sequence/string                                                         |
| **query sequence/string** | 検索時に呼び出し側から渡される sequence/string                                           |
| **corpus**                | data string の順序付き collection                                                        |
