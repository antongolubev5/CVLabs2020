"""
Задание:
Добрый вечер,

В приложении две лекции - obj detection и семантическая сегментация. Лабы решила всё же немного разнообразить.
1. Обучить разметке автодороги любую архитектуру из лекции
Данные:
https://www.dropbox.com/s/rrh8lrdclzlnxzv/full_CNN_train.p?dl=0
Разметка:
https://www.dropbox.com/s/ak850zqqfy6ily0/full_CNN_labels.p?dl=0
Желательно не только оценить meanIoU, но и возможность работать в реалтайм (те замерить, сколько идёт на  обработку одного изображения, и сколько можно паралелльно при этом обработать на одной видеокарте, скажем, 2080)
2. Обучиться на этом датасете. Тоже, выбираете ту архитектуру, что больше приглянулось, но желательно с минимальным обоснованием выбора https://groups.csail.mit.edu/vision/datasets/ADE20K/
3. Скорее object detection, но некоторые пытались эту задачу решить с помощью архитектур семантической сегментации и даже успешно. https://fki.tic.heia-fr.ch/databases/iam-handwriting-database - обучить находить строки или слова.

Одну задачу можно брать не более, чем трём людям.

Несколько важных заметок:
1. Помните про возможность аугументации (мы это делали в первой лабе)
2. Можно пользоваться предобученными моделями (Finetuning или Transfer Learning)
3. Я забыла это осветить в лекции, НО не во всех подходах один и тот же loss, так что перед реализацией архитектуры, сверьтесь со статьёй. Если не можете статью, заинтересовавшую вас нагуглить, обращайтесь, пришлю.

С уважением,
Олеся Криндач
"""


def main():
    pass


if __name__ == "__main__":
    main()