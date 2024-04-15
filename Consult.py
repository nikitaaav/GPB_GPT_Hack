import copy
import re
from typing import Any, Dict, List

from langchain.chains.base import Chain
from langchain.llms import BaseLLM
from langchain_core.prompts import ChatPromptTemplate

from deepinfra import ChatDeepInfra

llm = ChatDeepInfra(temperature=0.7)

from jsonanaliser import courses, names

class ConsultGPT(Chain):

    """Controller model for the Sales Agent."""
    course = courses
    names = names
    client = 'Клиент'
    current_conversation_stage = "1. Введение"
    salesperson_name = "Олег"
    salesperson_role = "специалист в сфере подбора курсов переквалификации в Газпромбанке"
    company_name = "Газпромбанк"
    company_business = "Газпромбанк - один из крупнейших универсальных банков России"
    job_vacancy = "консультант"
    conversation_type = "чат мессенджера"
    conversation_purpose = "Рекомендуй подходящий курс из списка. Выведи название и продолжительность выбранного курса."
    conversation_stage_dict = {
        "1": "Введение. Начни разговор с приветсвия и краткого представления себя {salesperson_role} и компании {company_name}. Поинтересуйся, какие курсы переквалификации интересны человеку. Пиши только на русском языке. Приветсвие должно состоять не больше чем из 45 слов. НЕ ПИШИ ВОСКЛИЦАТЕЛЬНЫЙ ЗНАК (!) И ЗАПЯТУЮ (,) ПЕРЕД ПРЕДЛОЖЕНИЕМ. ПЕРВОЕ СЛОВО В ПРЕДЛОЖЕНИИ С БОЛЬШОЙ БУКВЫ.",
        "2": "Ответ. Проанализировав ответ пользователя, на основании его ответа определи оптимальный курс из данного списка. Выведи ТОЛЬКО НАЗВАНИЕ курса ТОЛЬКО в квадратных скобках [] (название должно быть в списке {names}), без точек и знаков преринания. Больше ничего не пиши. Закончи диалог.",
    }

    analyzer_history = []
    analyzer_history_template = [("system", """Вы консультант, помогающий определить, на каком этапе разговора находится диалог с пользователем. Пиши ответы только на русском.

Определите, каким должен быть следующий непосредственный этап разговора о курсе, выбрав один из следующих вариантов:
1. Введение. Начни разговор с приветствия и краткого представления себя и компании, в которой ты работаешь. Попроси пользователя рассказать о том, чему он хочется научится. Пиши только на русском языке. Приветсвие должно состоять не больше чем из 45 слов. НЕ ПИШИ ВОСКЛИЦАТЕЛЬНЫЙ ЗНАК (!) И ЗАПЯТУЮ (,) ПЕРЕД ПРЕДЛОЖЕНИЕМ. ПЕРВОЕ СЛОВО В ПРЕДЛОЖЕНИИ С БОЛЬШОЙ БУКВЫ.
2. Ответ. Получив ответ от пользователя, определи наиболее оптимальный курс из данного списка. Выведи ТОЛЬКО НАЗВАНИЕ курса в квадратных скобках [] (название должно быть в списке {names}), без точек и знаков преринания. Больше ничего не пиши. Закончи диалог.""")]

    analyzer_system_postprompt_template = [("system", """Отвечайте только цифрой от 1 до 2, чтобы лучше понять, на каком этапе следует продолжить разговор.
Ответ должен состоять только из одной цифры, без слов.
Если истории разговоров нет, выведите 1.
Во всех остальных случаях, выведите 2.
Больше ничего не отвечайте и не добавляйте к своему ответу.

Текущая стадия разговора:
""")]

    conversation_history = []
    conversation_history_template = [("system", """Никогда не забывайте, что ваше имя {salesperson_name}, вы мужчина. Вы работаете {salesperson_role}. Вы работаете в компании под названием {company_name}. Бизнес {company_name} заключается в следующем: {company_business}.
Вот, что вы знаете о курсах:\n {course} \n. Названия курсов: {names}.
Название курса, должно быть из списка: {names}. Название курса должно быть выведено в квадартных скобках - []. 


Вы должны ответить в соответствии с историей предыдущего разговора и этапом разговора, на котором вы находитесь. Никогда не пишите информацию об этапе разговора.
Пиши, соблюдая все правила и нормы русского языка! Выводить только текст, без ссылок.

Вы ожидаете, что диалог будет выглядеть примерно следующим образом:
- Здравствуйте, меня зовут {salesperson_name}, я {job_vacancy} в компании {company_name}. Расскажите, чему примерно вы хотите научиться на курсах {company_name}?
- Здравствуйте. Хочу научиться языку программирования python.
- [Python-разработка]
""")]

    conversation_system_postprompt_template = [("system", """
НЕ ПИШИ ВОСКЛИЦАТЕЛЬНЫЙ ЗНАК И ЗАПЯТУЮ ПЕРЕД ПРЕДЛОЖЕНИЕМ. ПЕРВОЕ СЛОВО В ПРЕДЛОЖЕНИИ С БОЛЬШОЙ БУКВЫ.             
ПИШИ ТОЛЬКО ТЕКСТ!
Текущая стадия разговора:
{conversation_stage}

{salesperson_name}:
""")]

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    def retrieve_conversation_stage(self, key):
        return self.conversation_stage_dict.get(key, '1')

    def seed_agent(self):
        self.current_conversation_stage = self.retrieve_conversation_stage('1')
        self.analyzer_history = copy.deepcopy(self.analyzer_history_template)
        self.analyzer_history.append(("user", "Привет"))
        self.conversation_history = copy.deepcopy(self.conversation_history_template)
        self.conversation_history.append(("user", "Привет"))

    def human_step(self, human_message):
        self.analyzer_history.append(("user", human_message))
        self.conversation_history.append(("user", human_message))

    def ai_step(self):
        return self._call(inputs={})

    def analyse_stage(self):
        messages = self.analyzer_history + self.analyzer_system_postprompt_template
        template = ChatPromptTemplate.from_messages(messages)
#
        messages = template.format_messages(course=courses, names=self.names, salesperson_role=self.salesperson_role)
#
        response = llm.invoke(messages)
        conversation_stage_id = (re.findall(r'\b\d+\b', response.content) + ['1'])[0]

        self.current_conversation_stage = self.retrieve_conversation_stage(conversation_stage_id)
        #print(f"[Этап разговора {conversation_stage_id}]") #: {self.current_conversation_stage}")

    def _call(self, inputs: Dict[str, Any]) -> None:
        messages = self.conversation_history + self.conversation_system_postprompt_template
        template = ChatPromptTemplate.from_messages(messages)
        messages = template.format_messages(
            salesperson_name = self.salesperson_name,
            salesperson_role = self.salesperson_role,
            company_name = self.company_name,
            company_business = self.company_business,
            conversation_purpose = self.conversation_purpose,
            conversation_stage = self.current_conversation_stage,
            conversation_type = self.conversation_type,
            job_vacancy = self.job_vacancy,
            client = self.client,
            course = self.course,
            names = self.names
        )

        response = llm.invoke(messages)
        ai_message = (response.content).split('\n')[0]

        self.analyzer_history.append(("user", ai_message))
        self.conversation_history.append(("ai", ai_message))

        return ai_message

    @classmethod
    def from_llm(
        cls, llm: BaseLLM, verbose: bool = False, **kwargs
    ) -> "ConsultGPT":
        """Initialize the ConsultGPT Controller."""

        return cls(
            verbose = verbose,
            **kwargs,
        )