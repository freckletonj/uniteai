'''

Just getting the basic components of the State Loop Working

'''

import openai
import yaml
import os
import re

with open(os.path.expanduser('~/.uniteai.yml'), 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)
openai.api_key = cfg['openai']['api_key']

COMPLETION_ENGINES = [
    "text-davinci-003",
    "text-davinci-002",
    "ada",
    "babbage",
    "curie",
    "davinci",
]

CHAT_ENGINES = [
    "gpt-3.5-turbo",
    "gpt-4",
]

ENGINE = 'gpt-3.5-turbo'
# ENGINE = 'gpt-4'

def openai_autocomplete(engine, text, max_length):
    ''' NON-Streaming responses from OpenAI's API.'''
    if engine in COMPLETION_ENGINES:
        response = openai.Completion.create(
          engine=engine,
          prompt=text,
          max_tokens=max_length,
          stream=False
        )
        return response
    elif engine in CHAT_ENGINES:
        response = openai.ChatCompletion.create(
          model=engine,
          messages=[{"role": "user", "content": text}],
          stream=False
        )
        return response['choices'][0]['message']['content']


def find_tag(tag: str, doc_lines: [str]):
    ''' Find index of first element that contains `tag`. '''
    ix = 0
    for ix, line in enumerate(doc_lines):
        match = re.search(tag, line)
        if match:
            return ix, match.start(), match.end()
    return None


def find_block(start_tag, end_tag, doc):
    '''Fine the indices of a start/end-tagged block.'''
    if doc is None:
        return None, None
    doc_lines = doc.split('\n')
    s = find_tag(start_tag, doc_lines)
    e = find_tag(end_tag, doc_lines)
    return s, e


def extract_block(start, end, doc):
    '''Extract block of text between `start` and `end` tag.'''
    if doc is None:
        return None
    doc_lines = doc.split('\n')
    if start is None or end is None:
        return None
    if start[0] > end[0] or (start[0] == end[0] and start[2] > end[1]):
        return None
    if start[0] == end[0]:
        return [doc_lines[start[0]][start[2]: end[1]]]
    else:
        block = [doc_lines[start[0]][start[2]:]]  # portion of start line
        block.extend(doc_lines[start[0]+1:end[0]])  # all of middle lines
        block.append(doc_lines[end[0]][:end[1]])  # portion of end line
        return '\n'.join(block)


def start_tag(x):
    return f'<{x}_TAG>'


def end_tag(x):
    return f'</{x}_TAG>'


def get_block(tag, doc):
    s1, s2 = find_block(start_tag(tag), end_tag(tag), doc)
    return extract_block(s1, s2, doc)


STATE = 'STATE'
NEW_STATE = 'NEW_STATE'
REQUEST = 'REQUEST'
RESPONSE = 'RESPONSE'
UPDATES_NEEDED = 'UPDATES_NEEDED'

state = '''
players:
  josh:
    items:
    location:
  kirtley:
    items:
    location:

quests:

obstacles:

enemies:
'''

def get_response(request,
                 running_resp,
                 state,
                 prefix=None,
                 suffix=None):
    nl = '\n\n'  # can't do newlines inside f-exprs
    prompt = f'''
{prefix + nl if prefix else ''}You must assume the role of a finite state machine, but using only natural language.

You will be given state, and a request.

You must return a response, and a new state.

Please format your response like:

{start_tag(RESPONSE)}
your response
{end_tag(RESPONSE)}

{start_tag(UPDATES_NEEDED)}
updates that you'll need to apply to the new state
{end_tag(UPDATES_NEEDED)}

{start_tag(NEW_STATE)}
the new state
{end_tag(NEW_STATE)}

Here is the current state:

{start_tag(STATE)}
{state}
{end_tag(STATE)}

Here is a transcript of your responses so far:
{running_resp}

Here is the current request:
    {request}{nl + suffix if suffix else ''}
'''.strip()

    return openai_autocomplete(ENGINE, prompt, max_length=200)

prefix = 'You will be a Dungeon Master, and you will keep notes via a natural language-based state machine. Keep notes on: items, players, quests, etc.'
suffix = 'Remember, keep responses brief, invent interesting quests and obstacles, and make sure the state is always accurate.'

print('Welcome!')

running_resp = ''
while True:
    request = input('Your Command:')
    x = get_response(request,
                     running_resp=running_resp,
                     state=state,
                     prefix=prefix,
                     suffix=suffix)

    # Try extracting new_state
    new_state = get_block(NEW_STATE, x)
    if new_state is None:
        new_state = get_block(STATE, x)

    # Try extracting response
    resp = get_block(RESPONSE, x)
    if resp is None:
        resp = '<AI response invalid>'
        print(f'INVALID RESPONSE: \n{x}')
        continue

    if new_state is not None and resp is not None:
        state = new_state
        print(f'STATE: {state}')
        running_resp = f'{running_resp.strip()}\n\n{resp.strip()}'
        print(f'RESPONSE: {running_resp}')
