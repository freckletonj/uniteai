'''

`Edits` is awkward to test since it intrinsically deals with concurrency. So,
these tests are ugly.

'''

from llmpal.edit import Edits
import threading
import re
import time
from dataclasses import dataclass


##################################################
# Test Helper


##########
# Test Datatype

@dataclass
class FakeJob:
    start_tag: str
    end_tag: str
    text: str
    strict: str  # strict jobs MUST be applied, non-strict may be skipped (eg
    should_fail: bool  # useful for testing


##########
# Replace

def replace_between_tags(document, start_tag, end_tag, text):
    pattern = f'{start_tag}(.*?){end_tag}'
    new_text = re.sub(pattern,
                      lambda m: f'{start_tag}{text}{end_tag}',
                      document)
    return new_text


def test_replace_between_tags():
    doc = 'Hello <tag>World</tag>, how are you?'
    replacement = "Everyone"
    start_tag = "<tag>"
    end_tag = "</tag>"
    new_doc = replace_between_tags(doc, start_tag, end_tag, replacement)
    assert new_doc == 'Hello <tag>Everyone</tag>, how are you?'


##########
# Append

def append_between_tags(document, start_tag, end_tag, text):
    pattern = f'{start_tag}(.*?){end_tag}'
    match = re.findall(pattern, document)[0]
    new_text = re.sub(pattern,
                      lambda m: f'{start_tag}{match}{text}{end_tag}',
                      document)
    return new_text


def test_append_between_tags():
    doc = 'Hello <tag>World</tag>, how are you?'
    addition = "Everyone"
    start_tag = "<tag>"
    end_tag = "</tag>"
    new_doc = append_between_tags(doc, start_tag, end_tag, addition)
    assert new_doc == 'Hello <tag>WorldEveryone</tag>, how are you?'


##########
# Threadsafe var

class ThreadSafeVar:
    '''
    A threadsafe variable
    '''

    def __init__(self, initial):
        self.value = initial
        self._lock = threading.Lock()

    def set(self, x):
        with self._lock:
            self.value = x
            return self.value

    def modify(self, f):
        with self._lock:
            self.value = f(self.value)
            return self.value

    def get(self):
        return self.value


##################################################
# Tests

# a fixture
document = """
:START_A:ORIG_ONE:END_A:
:START_B:ORIG_TWO:END_B:
"""


def replace_in_doc(job: FakeJob, doc: ThreadSafeVar):
    ''' mutate the var by replacing between tags '''
    doc.modify(lambda orig:
               replace_between_tags(orig,
                                    job.start_tag,
                                    job.end_tag,
                                    job.text))


def append_in_doc(job: FakeJob, doc: ThreadSafeVar):
    ''' mutate the var by appending between tags '''
    doc.modify(lambda orig:
               append_between_tags(orig,
                                   job.start_tag,
                                   job.end_tag,
                                   job.text))


def applicator_fn(f, threadsafe_var):
    ''' a function builder that can simulate job failure '''
    def applicator_fn(job):
        if job.should_fail:  # simulate failure to apply edit
            return False
        f(job, threadsafe_var)
        return True  # job succeeded
    return applicator_fn


def replace_applicator_fn(var):
    return applicator_fn(replace_in_doc, var)


def append_applicator_fn(var):
    return applicator_fn(append_in_doc, var)


def mk_job(tag, text, strict, should_fail):
    ''' a `Job` builder. '''
    return FakeJob(
        start_tag=f':START_{tag}:',
        end_tag=f':END_{tag}:',
        text=text,
        strict=strict,
        should_fail=should_fail
    )


def wait(q):
    '''This is a little hacky, but, block for queues to be empty'''
    while not q.empty():
        time.sleep(0.01)


def test_1():
    n = 'test_queue'
    doc = ThreadSafeVar(document[:])
    app_fn = replace_applicator_fn(doc)
    edits = Edits(app_fn)
    edits.create_job_queue(n)
    edits.start()
    edits.add_job(n, mk_job('A', 'JOB A', strict=True, should_fail=False))
    edits.add_job(n, mk_job('B', 'JOB B', strict=True, should_fail=False))
    wait(edits.edit_jobs[n])
    expected = """
:START_A:JOB A:END_A:
:START_B:JOB B:END_B:
"""
    assert doc.get() == expected

def test_2():
    n = 'test_queue'
    doc = ThreadSafeVar(document[:])
    app_fn = replace_applicator_fn(doc)
    edits = Edits(app_fn)
    edits.create_job_queue(n)
    edits.start()
    edits.add_job(n, mk_job('A', 'JOB A', strict=True, should_fail=False))
    edits.add_job(n, mk_job('B', 'JOB B', strict=True, should_fail=True))
    wait(edits.edit_jobs[n])
    # NOTE: this can finish even though a `strict` failed because we only wait
    # for the queue to empty, not for all failed jobs to also eventually
    # succeed.
    expected = """
:START_A:JOB A:END_A:
:START_B:ORIG_TWO:END_B:
"""
    assert doc.get() == expected


def test_3():
    n = 'test_queue'
    doc = ThreadSafeVar(document[:])
    app_fn = replace_applicator_fn(doc)
    edits = Edits(app_fn)
    edits.create_job_queue(n)
    edits.start()
    edits.add_job(n, mk_job('A', 'JOB A', strict=False, should_fail=True))
    edits.add_job(n, mk_job('B', 'JOB B', strict=True, should_fail=False))
    wait(edits.edit_jobs[n])
    expected = """
:START_A:ORIG_ONE:END_A:
:START_B:JOB B:END_B:
"""
    assert doc.get() == expected


def test_4_same_queue():
    n = 'test_queue'
    doc = ThreadSafeVar(document[:])
    app_fn = append_applicator_fn(doc)
    edits = Edits(app_fn)
    edits.create_job_queue(n)
    edits.add_job(n, mk_job('A', '1', strict=False, should_fail=False))
    edits.add_job(n, mk_job('A', '2', strict=False, should_fail=False))
    edits.add_job(n, mk_job('A', '3', strict=False, should_fail=False))
    edits.add_job(n, mk_job('A', '4', strict=False, should_fail=False))
    edits.add_job(n, mk_job('B', 'x', strict=False, should_fail=False))
    edits.start()
    wait(edits.edit_jobs[n])
    expected = """
:START_A:ORIG_ONE:END_A:
:START_B:ORIG_TWOx:END_B:
"""
    assert doc.get() == expected


def test_5_different_queue():
    doc = ThreadSafeVar(document[:])
    app_fn = append_applicator_fn(doc)
    edits = Edits(app_fn)
    edits.create_job_queue('qA')
    edits.create_job_queue('qB')
    edits.add_job('qA', mk_job('A', '1', strict=False, should_fail=False))
    edits.add_job('qA', mk_job('A', '2', strict=False, should_fail=False))
    edits.add_job('qA', mk_job('A', '3', strict=False, should_fail=False))
    edits.add_job('qA', mk_job('A', '4', strict=False, should_fail=False))
    edits.add_job('qB', mk_job('B', 'x', strict=False, should_fail=False))
    edits.start()
    wait(edits.edit_jobs['qA'])
    wait(edits.edit_jobs['qB'])
    expected = """
:START_A:ORIG_ONE4:END_A:
:START_B:ORIG_TWOx:END_B:
"""
    assert doc.get() == expected


def test_6_strict():
    doc = ThreadSafeVar(document[:])
    app_fn = append_applicator_fn(doc)
    edits = Edits(app_fn)
    edits.create_job_queue('qA')
    edits.create_job_queue('qB')
    edits.add_job('qA', mk_job('A', '1', strict=True, should_fail=False))
    edits.add_job('qA', mk_job('A', '2', strict=True, should_fail=False))
    edits.add_job('qA', mk_job('A', '3', strict=True, should_fail=False))
    edits.add_job('qA', mk_job('A', '4', strict=True, should_fail=False))
    edits.add_job('qB', mk_job('B', 'x', strict=False, should_fail=False))
    edits.add_job('qB', mk_job('B', 'y', strict=False, should_fail=False))
    edits.add_job('qB', mk_job('B', 'z', strict=False, should_fail=False))
    edits.start()
    wait(edits.edit_jobs['qA'])
    wait(edits.edit_jobs['qB'])
    expected = """
:START_A:ORIG_ONE1234:END_A:
:START_B:ORIG_TWOz:END_B:
"""
    assert doc.get() == expected


def test_7_non_strict_but_pauses():
    ''' Non strict append-edits but they have time to apply before getting
    passed over. '''
    pause = 0.3
    doc = ThreadSafeVar(document[:])
    app_fn = append_applicator_fn(doc)
    edits = Edits(app_fn)
    edits.create_job_queue('qA')
    edits.create_job_queue('qB')
    edits.start()
    time.sleep(pause)
    edits.add_job('qA', mk_job('A', '1', strict=False, should_fail=False))
    time.sleep(pause)
    edits.add_job('qA', mk_job('A', '2', strict=False, should_fail=False))
    time.sleep(pause)
    edits.add_job('qA', mk_job('A', '3', strict=False, should_fail=False))
    time.sleep(pause)
    edits.add_job('qA', mk_job('A', '4', strict=False, should_fail=False))
    edits.add_job('qB', mk_job('B', 'x', strict=False, should_fail=False))
    edits.add_job('qB', mk_job('B', 'y', strict=False, should_fail=False))
    edits.add_job('qB', mk_job('B', 'z', strict=False, should_fail=False))
    wait(edits.edit_jobs['qA'])
    wait(edits.edit_jobs['qB'])
    expected = """
:START_A:ORIG_ONE1234:END_A:
:START_B:ORIG_TWOz:END_B:
"""
    assert doc.get() == expected
