# Intro
AnswerBot is published in 'AnswerBot: Automated generation of answer summary to developers’ technical questions' 2017 ASE
# Environment
to run answerbot, you need to active an vitural environment in which the python version is 2.x. After that, you need to run
```
pip install -r requirements.txt
```

# Pipeline
to run answerbot, you need to run
```
src/main.py
```
the result is stored in 
```
result
```
folder.

Also, you can use 
```
rouge_evaluate.py
```
to evaluate the performance of Lexrank.
## Rouge Setting
To set the rouge environment, please follow [this instruction](https://stackoverflow.com/a/57686103/10143020)
