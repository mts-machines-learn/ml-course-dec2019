# 5. Линейная регрессия. Часть 2.

В этой лекции мы продолжим обсуждать линейную регрессию. Мы применим ее в реальной задаче. Также мы обсудим полиномиальную регрессию. Узнаем, как бороться с переобучением и обсудим среднюю абсолютную ошибку.

[📺 Трансляция](https://youtu.be/mOoZI4cHwFU) 17 декабря в 16:00 по Москве.

[📒 Лекция](https://github.com/mts-machines-learn/ml-course-dec2019/blob/master/5.%20%D0%9B%D0%B8%D0%BD%D0%B5%D0%B9%D0%BD%D0%B0%D1%8F%20%D1%80%D0%B5%D0%B3%D1%80%D0%B5%D1%81%D1%81%D0%B8%D1%8F%20ll/Regression_2.ipynb) <a href="https://github.com/mts-machines-learn/ml-course-dec2019/blob/master/5.%20%D0%9B%D0%B8%D0%BD%D0%B5%D0%B9%D0%BD%D0%B0%D1%8F%20%D1%80%D0%B5%D0%B3%D1%80%D0%B5%D1%81%D1%81%D0%B8%D1%8F%20ll/Regression_2.ipynb"></a>


### Домашнее задание

Домашнее задание в файле [Homework_5.ipynb](https://github.com/mts-machines-learn/ml-course-dec2019/blob/master/5.%20%D0%9B%D0%B8%D0%BD%D0%B5%D0%B9%D0%BD%D0%B0%D1%8F%20%D1%80%D0%B5%D0%B3%D1%80%D0%B5%D1%81%D1%81%D0%B8%D1%8F%20ll/Homework_5.ipynb). Вы можете клонировать репозиторий к себе на диск и сделать домашку в локальном Jupyter Notebook, или можете открыть файл в Colab: <a href="https://colab.research.google.com/github/mts-machines-learn/ml-course-dec2019/blob/master/5.%20%D0%9B%D0%B8%D0%BD%D0%B5%D0%B9%D0%BD%D0%B0%D1%8F%20%D1%80%D0%B5%D0%B3%D1%80%D0%B5%D1%81%D1%81%D0%B8%D1%8F%20ll/Homework_5.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a>.

**Внимание!** Для корректной работы в Colab необходимо загрузить в него файл [regression2_helper.py](https://raw.githubusercontent.com/mts-machines-learn/ml-course-dec2019/master/5.%20%D0%9B%D0%B8%D0%BD%D0%B5%D0%B9%D0%BD%D0%B0%D1%8F%20%D1%80%D0%B5%D0%B3%D1%80%D0%B5%D1%81%D1%81%D0%B8%D1%8F%20ll/regression2_helper.py). Для этого откройте панель слева нажатием на стрелочку`→Files→Upload`.

После того, как вы закончите упражнения, сохраните файл. Если вы делаете домашку в Colab, скачайте файл с помощью пункта меню `File → Download *.ipynb`.
