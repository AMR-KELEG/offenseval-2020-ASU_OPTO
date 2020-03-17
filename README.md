# offenseval-2020-ASU_OPTO
This repository contains the models that was developed as part of the OffenseEval 2020 competition for Arabic organized by [**SemEval-2020**](http://alt.qcri.org/semeval2020/index.php?id=tasks) and [**OSACT4**](http://edinburghnlp.inf.ed.ac.uk/workshops/OSACT4/)

# Tasks description
The Original task was **Offensive text detection**. Although there might not be a definition that is agreed upon by researchers, the following definition seems to be the one that is used by the task's organizers (Any text that might be considered as inappropriate is offensive. This includes profanity, hate speech or toxic comments.)
For the OSACT4 competition, another subtask was proposed and aimed at Hate speech detection (Offensive text targeting a person or a group of people). This is harder in general than detecting offensive text.

# Results
**Model name** | Accuracy (train) | Precision (train)| Recall (train)| F1 (train)| Accuracy (dev)| Precision (dev)| Recall (dev)| F1 (dev)
-- | -- | -- | -- | -- | -- | -- | -- | --
tfidf + logistic regression | 0.889 | 0.938 | 0.725 | 0.778 | 0.888 | **0.921** | 0.694 | 0.746
CNN + Aravec | 0.982 | 0.985 | 0.959 | 0.971 | 0.928 | 0.906 | 0.838 | 0.867
BiLSTM | 0.999 | 0.998 | 0.998 | 0.998 | 0.920 | 0.856 | **0.884** | 0.869
Multi-lingual BERT | 0.978 | 0.975 | 0.956 | 0.965 | 0.905 | 0.855 | 0.805 | 0.826
AraBERT | 0.998 | 0.998 | 0.994 | 0.996 | **0.928** | 0.881 | 0.871 | **0.876**

# How to use
Each model is developed as a separate jupyter notebook. You might need to use Google Colab and upload the data so that you can use GPUs.
