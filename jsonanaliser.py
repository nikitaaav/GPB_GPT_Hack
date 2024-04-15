import json

with open('C:\\Users\\Никита\\Documents\\Проекты\\GPB_GPT_Hack\\course.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)
    courses = ''
    names = ''
    k = 1
    for e in data:
        courses += f'{k}. Название курса: {e["Course_name"]}. '
        courses += f'Длительность: {e["Duration"]}. '
        courses += f'Описание: {e["Description"]}. '
        courses += f'Что будете извучать: {e["What_you_will_learn"]}. '
        courses += f'Программа курса: {e["Course_program"]}. '
        courses += '\n '
        names += f' {e["Course_name"]},'
        k += 1
