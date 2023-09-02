'''

DEPS:
pip install lark-parser

'''

from jinja2 import Template
print()

def generate_llama2_prompt(system_message, conversation):
    '''
    Generates a prompt for Llama 2 chat models based on the given system message and list of conversation elements.

    Parameters:
    - system_message (str): The system message to instruct the model.
    - conversation (list of dicts): A list containing dictionaries with 'isUser' and 'text' keys.

    Returns:
    - str: The complete prompt formatted for Llama 2 models.
    '''

    template_str = '''<s>[INST] <<SYS>>
{{ system_message }}
<</SYS>>'''

    for idx, message in enumerate(conversation):
        if message["isUser"]:
            template_str += '''
{{ conversation[''' + str(idx) + '''].text }} [/INST]'''
        else:
            template_str += ''' {{ conversation[''' + str(idx) + '''].text }} </s><s>[INST]'''

    template = Template(template_str)
    prompt = template.render(system_message=system_message, conversation=conversation)

    return prompt

# Example usage:

system_message = "The System"
conversation = [
    {"isUser": True, "text": "User 1"},
    {"isUser": False, "text": "Response 1"},
]

prompt = generate_llama2_prompt(system_message, conversation)
# print(prompt)


##################################################
# Parser

from lark import Lark, Transformer, v_args

grammar = r"""
    start: block*

    block: open_tag optional_user text close_tag  -> closed_block
         | open_tag optional_user text?           -> unclosed_block

    open_tag: "<s>"
    close_tag: "</s>"

    optional_sys: "<<SYS>>" text "<</SYS>>"?
    optional_user: "[INST]" optional_sys text "[/INST]"?

    text: /(?!<\/s>|\[\/INST\])[^<\[]+/

    %ignore /\s+/
"""

@v_args(inline=True)
class MyTransformer(Transformer):
    def start(self, *args):
        return list(args)

    def closed_block(self, ot, user, response, ct):
        sys = user['sys']
        req = user['req']
        return {'sys':sys, 'req':req, 'response':response.value}

    def unclosed_block(self, *args):
        return {"block": self.create_sub_dict(args), "UNCLOSED": True}

    def optional_sys(self, text):
        return text.value

    def optional_user(self, optional_sys, text):
        return {"sys": optional_sys, "req": text.value}

    def text(self, text):
        return text

    @staticmethod
    def create_sub_dict(args):
        sub_dict = {}
        for arg in args:
            if isinstance(arg, dict):
                sub_dict.update(arg)
            else:
                sub_dict["text"] = arg
        return sub_dict


parser = Lark(grammar, start='start', parser='lalr', transformer=MyTransformer())

text = """
<s> [INST]
<<SYS>>hello<</SYS>>
dear[/INST]
world
</s>

<s> [INST]
<<SYS>>hello<</SYS>>
dear[/INST]
""".strip()

result = parser.parse(text)
print(result)
