'''

`Edits` is awkward to test since it intrinsically deals with concurrency. So,
these tests are ugly. I create a mocked version of what the LSP is doing, which
is essentially just a string document that can be accessed threadsafely.

'''

from uniteai.edit import Edits
import threading
import re
import time
from dataclasses import dataclass
import concurrent.futures

# TODO: REINCORPORATE THESE TESTS, since removing the Assistant concept
# (superceded by proper Actor Models, via Thespian)




# from uniteai.assistant import Assistant
# from typing import Union
# import pytest
# from uniteai.common import insert_text_at

# JOB_DELAY = 0.001


# ##################################################
# # Test Helper

# ##########
# # Test Datatype


# @dataclass
# class FakeInsertJob:
#     text: str
#     line: int
#     column: int
#     strict: bool
#     should_fail: bool

# @dataclass
# class FakeBlockJob:
#     start_tag: str
#     end_tag: str
#     text: str
#     strict: bool  # strict jobs MUST be applied, non-strict may be skipped (eg
#     should_fail: bool  # useful for testing


# @dataclass
# class FakeDeleteJob:
#     regex: str
#     strict: bool  # strict jobs MUST be applied, non-strict may be skipped (eg
#     should_fail: bool  # useful for testing


# FakeJob = Union[FakeInsertJob, FakeBlockJob, FakeDeleteJob]


# ##########
# # Replace Between Tags

# def replace_between_tags(document, start_tag, end_tag, text):
#     pattern = f'{start_tag}(.*?){end_tag}'
#     new_text = re.sub(pattern,
#                       lambda m: f'{start_tag}{text}{end_tag}',
#                       document)
#     return new_text


# def test_replace_between_tags():
#     doc = 'Hello <tag>World</tag>, how are you?'
#     replacement = "Everyone"
#     start_tag = "<tag>"
#     end_tag = "</tag>"
#     new_doc = replace_between_tags(doc, start_tag, end_tag, replacement)
#     assert new_doc == 'Hello <tag>Everyone</tag>, how are you?'


# ##########
# # Append

# def append_between_tags(document, start_tag, end_tag, text):
#     pattern = f'{start_tag}(.*?){end_tag}'
#     match = re.findall(pattern, document)[0]
#     new_text = re.sub(pattern,
#                       lambda m: f'{start_tag}{match}{text}{end_tag}',
#                       document)
#     return new_text


# def test_append_between_tags():
#     doc = 'Hello <tag>World</tag>, how are you?'
#     addition = "Everyone"
#     start_tag = "<tag>"
#     end_tag = "</tag>"
#     new_doc = append_between_tags(doc, start_tag, end_tag, addition)
#     assert new_doc == 'Hello <tag>WorldEveryone</tag>, how are you?'


# ##########
# # Threadsafe var

# class ThreadSafeVar:
#     '''
#     A threadsafe variable
#     '''

#     def __init__(self, initial):
#         self.value = initial
#         self._lock = threading.Lock()

#     def set(self, x):
#         with self._lock:
#             self.value = x
#             return self.value

#     def modify(self, f):
#         with self._lock:
#             self.value = f(self.value)
#             return self.value

#     def get(self):
#         return self.value


# ##################################################
# # Tests

# # a fixture
# document = """
# :START_A:ORIG_ONE:END_A:
# :START_B:ORIG_TWO:END_B:
# """


# def replace_in_doc(job: FakeJob, doc: ThreadSafeVar):
#     ''' mutate the var by replacing between tags '''
#     doc.modify(lambda orig:
#                replace_between_tags(orig,
#                                     job.start_tag,
#                                     job.end_tag,
#                                     job.text))


# def append_in_doc(job: FakeJob, doc: ThreadSafeVar):
#     ''' mutate the var by appending between tags '''
#     if isinstance(job, FakeInsertJob):
#         def i(orig):
#             return insert_text_at(orig, job.text, job.line, job.column)
#         doc.modify(i)

#     if isinstance(job, FakeBlockJob):
#         doc.modify(lambda orig:
#                    append_between_tags(orig,
#                                        job.start_tag,
#                                        job.end_tag,
#                                        job.text))
#     if isinstance(job, FakeDeleteJob):
#         def r(orig):
#             return re.sub(job.regex, '', orig)
#         doc.modify(r)


# def applicator_fn(f, threadsafe_var):
#     ''' a function builder that can simulate job failure '''
#     def applicator_fn(job):
#         if job.should_fail:  # simulate failure to apply edit
#             return False
#         f(job, threadsafe_var)
#         return True  # job succeeded
#     return applicator_fn


# def replace_applicator_fn(var):
#     return applicator_fn(replace_in_doc, var)


# def append_applicator_fn(var):
#     return applicator_fn(append_in_doc, var)


# def mk_job(tag, text, strict, should_fail):
#     ''' a `Job` builder. '''
#     return FakeBlockJob(
#         start_tag=f':START_{tag}:',
#         end_tag=f':END_{tag}:',
#         text=text,
#         strict=strict,
#         should_fail=should_fail
#     )


# def wait(q):
#     '''This is a little hacky, but, block for queues to be empty'''
#     while not q.empty():
#         time.sleep(0.01)


# def test_1():
#     n = 'test_queue'
#     doc = ThreadSafeVar(document[:])
#     app_fn = replace_applicator_fn(doc)
#     edits = Edits(app_fn, job_delay=JOB_DELAY)
#     edits.create_job_queue(n)
#     edits.start()
#     edits.add_job(n, mk_job('A', 'JOB A', strict=True, should_fail=False))
#     edits.add_job(n, mk_job('B', 'JOB B', strict=True, should_fail=False))
#     wait(edits.edit_jobs[n])
#     expected = """
# :START_A:JOB A:END_A:
# :START_B:JOB B:END_B:
# """
#     assert doc.get() == expected

# def test_2():
#     n = 'test_queue'
#     doc = ThreadSafeVar(document[:])
#     app_fn = replace_applicator_fn(doc)
#     edits = Edits(app_fn, job_delay=JOB_DELAY)
#     edits.create_job_queue(n)
#     edits.start()
#     edits.add_job(n, mk_job('A', 'JOB A', strict=True, should_fail=False))
#     # edits.add_job(n, mk_job('B', 'JOB B', strict=True, should_fail=True))
#     wait(edits.edit_jobs[n])
#     # NOTE: this can finish even though a `strict` failed because we only wait
#     # for the queue to empty, not for all failed jobs to also eventually
#     # succeed.
#     expected = """
# :START_A:JOB A:END_A:
# :START_B:ORIG_TWO:END_B:
# """
#     assert doc.get() == expected


# def test_3():
#     n = 'test_queue'
#     doc = ThreadSafeVar(document[:])
#     app_fn = replace_applicator_fn(doc)
#     edits = Edits(app_fn, job_delay=JOB_DELAY)
#     edits.create_job_queue(n)
#     edits.start()
#     edits.add_job(n, mk_job('A', 'JOB A', strict=False, should_fail=True))
#     edits.add_job(n, mk_job('B', 'JOB B', strict=True, should_fail=False))
#     wait(edits.edit_jobs[n])
#     expected = """
# :START_A:ORIG_ONE:END_A:
# :START_B:JOB B:END_B:
# """
#     assert doc.get() == expected


# def test_4_same_queue():
#     n = 'test_queue'
#     doc = ThreadSafeVar(document[:])
#     app_fn = append_applicator_fn(doc)
#     edits = Edits(app_fn, job_delay=JOB_DELAY)
#     edits.create_job_queue(n)
#     edits.add_job(n, mk_job('A', '1', strict=False, should_fail=False))
#     edits.add_job(n, mk_job('A', '2', strict=False, should_fail=False))
#     edits.add_job(n, mk_job('A', '3', strict=False, should_fail=False))
#     edits.add_job(n, mk_job('A', '4', strict=False, should_fail=False))
#     edits.add_job(n, mk_job('B', 'x', strict=False, should_fail=False))
#     edits.start()
#     wait(edits.edit_jobs[n])
#     expected = """
# :START_A:ORIG_ONE:END_A:
# :START_B:ORIG_TWOx:END_B:
# """
#     assert doc.get() == expected


# def test_5_different_queue():
#     doc = ThreadSafeVar(document[:])
#     app_fn = append_applicator_fn(doc)
#     edits = Edits(app_fn, job_delay=JOB_DELAY)
#     edits.create_job_queue('qA')
#     edits.create_job_queue('qB')
#     edits.add_job('qA', mk_job('A', '1', strict=False, should_fail=False))
#     edits.add_job('qA', mk_job('A', '2', strict=False, should_fail=False))
#     edits.add_job('qA', mk_job('A', '3', strict=False, should_fail=False))
#     edits.add_job('qA', mk_job('A', '4', strict=False, should_fail=False))
#     edits.add_job('qB', mk_job('B', 'x', strict=False, should_fail=False))
#     edits.start()
#     wait(edits.edit_jobs['qA'])
#     wait(edits.edit_jobs['qB'])
#     expected = """
# :START_A:ORIG_ONE4:END_A:
# :START_B:ORIG_TWOx:END_B:
# """
#     assert doc.get() == expected


# def test_6_strict():
#     doc = ThreadSafeVar(document[:])
#     app_fn = append_applicator_fn(doc)
#     edits = Edits(app_fn, job_delay=JOB_DELAY)
#     edits.create_job_queue('qA')
#     edits.create_job_queue('qB')
#     edits.add_job('qA', mk_job('A', '1', strict=True, should_fail=False))
#     edits.add_job('qA', mk_job('A', '2', strict=True, should_fail=False))
#     edits.add_job('qA', mk_job('A', '3', strict=True, should_fail=False))
#     edits.add_job('qA', mk_job('A', '4', strict=True, should_fail=False))
#     edits.add_job('qB', mk_job('B', 'x', strict=False, should_fail=False))
#     edits.add_job('qB', mk_job('B', 'y', strict=False, should_fail=False))
#     edits.add_job('qB', mk_job('B', 'z', strict=False, should_fail=False))
#     edits.start()
#     wait(edits.edit_jobs['qA'])
#     wait(edits.edit_jobs['qB'])
#     expected = """
# :START_A:ORIG_ONE1234:END_A:
# :START_B:ORIG_TWOz:END_B:
# """
#     assert doc.get() == expected


# def test_7_non_strict_but_pauses():
#     ''' Non strict append-edits but they have time to apply before getting
#     passed over. '''
#     m = 1.5  # multiple to increase pause by
#     doc = ThreadSafeVar(document[:])
#     app_fn = append_applicator_fn(doc)
#     edits = Edits(app_fn, job_delay=JOB_DELAY)
#     edits.create_job_queue('qA')
#     edits.create_job_queue('qB')
#     edits.start()
#     time.sleep(JOB_DELAY * m)
#     edits.add_job('qA', mk_job('A', '1', strict=False, should_fail=False))
#     time.sleep(JOB_DELAY * m)
#     edits.add_job('qA', mk_job('A', '2', strict=False, should_fail=False))
#     time.sleep(JOB_DELAY * m)
#     edits.add_job('qA', mk_job('A', '3', strict=False, should_fail=False))
#     time.sleep(JOB_DELAY * m)
#     edits.add_job('qA', mk_job('A', '4', strict=False, should_fail=False))
#     edits.add_job('qB', mk_job('B', 'x', strict=False, should_fail=False))
#     edits.add_job('qB', mk_job('B', 'y', strict=False, should_fail=False))
#     edits.add_job('qB', mk_job('B', 'z', strict=False, should_fail=False))
#     wait(edits.edit_jobs['qA'])
#     wait(edits.edit_jobs['qB'])
#     expected = """
# :START_A:ORIG_ONE1234:END_A:
# :START_B:ORIG_TWOz:END_B:
# """
#     assert doc.get() == expected


# ##################################################
# # Assistant Tests (TODO: move these to own module)

# def streaming_function(name, start_tag, end_tag, start_i, end_i):
#     ''' count up to end_i, and stream edits to a block '''
#     def f(should_stop_event, edits, i=start_i):
#         for i in range(start_i, end_i):
#             if should_stop_event.is_set():
#                 break
#             edits.add_job(name,
#                           FakeBlockJob(
#                               start_tag=start_tag,
#                               end_tag=end_tag,
#                               text=str(i),
#                               strict=True,
#                               should_fail=False,
#                           ))
#             time.sleep(JOB_DELAY*10)
#     return f


# def init_function(name, text, line, col):
#     def f(edits):
#         edits.add_job(name,
#                       FakeInsertJob(f':START_{name}::END_{name}:',
#                                     line, col, strict=True, should_fail=False))
#     return f


# def cleanup_function(name, start_tag, end_tag):
#     def f(edits):
#         edits.add_job(name,
#                       FakeDeleteJob(
#                           regex=start_tag,
#                           strict=True,
#                           should_fail=False,
#                       ))
#         edits.add_job(name,
#                       FakeDeleteJob(
#                           regex=end_tag,
#                           strict=True,
#                           should_fail=False,
#                       ))
#     return f


# def on_start_but_running(n):
#     def f():
#         raise Exception(f'{n} is already running')
#     return f


# def on_stop_but_stopped(n):
#     def f():
#         raise Exception(f'{n} is already stopped')
#     return f


# def test_concurrent_assistants_explicit_stopping():
#     '''
#     Test where threads are explicitly stopped
#     '''
#     executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

#     doc = ThreadSafeVar('')
#     app_fn = append_applicator_fn(doc)
#     edits = Edits(app_fn, job_delay=JOB_DELAY)

#     n1 = "A"
#     start_tag_1 = ':START_A:'
#     end_tag_1 = ':END_A:'
#     assistant1 = Assistant(
#         name=n1,
#         executor=executor,
#         edits=edits,
#         on_start_but_running=on_start_but_running(n1),
#         on_stop_but_stopped=on_stop_but_stopped(n1),
#     )

#     n2 = "B"
#     start_tag_2 = ':START_B:'
#     end_tag_2 = ':END_B:'
#     assistant2 = Assistant(
#         name=n2,
#         executor=executor,
#         edits=edits,
#         on_start_but_running=on_start_but_running(n2),
#         on_stop_but_stopped=on_stop_but_stopped(n2),
#     )

#     # Create job queues for each assistant
#     edits.create_job_queue(n1)
#     edits.create_job_queue(n2)

#     # Start the edits thread
#     edits.start()

#     # Start Assistant1
#     assistant1.start(init_function(n1, n1, 1, -1),
#                      streaming_function(n1, start_tag_1, end_tag_1, 1, int(1e6)),
#                      cleanup_function(n1, start_tag_1, end_tag_1))
#     with pytest.raises(Exception) as e:
#         assistant1.start(None, None, None)
#     assert str(e.value) == 'A is already running'

#     # Start Assistant2
#     assistant2.start(init_function(n2, n2, 2, -1),
#                      streaming_function(n2, start_tag_2, end_tag_2, 1001, int(1e6)),
#                      cleanup_function(n2, start_tag_2, end_tag_2))
#     with pytest.raises(Exception) as e:
#         assistant2.start(None, None, None)
#     assert str(e.value) == 'B is already running'

#     # Let them run for 10x the delay streaming_function
#     time.sleep(JOB_DELAY*100)

#     # Stop the assistants
#     with pytest.raises(Exception) as e:
#         assistant1.stop()
#         assistant1.stop()  # stop twice
#     assert str(e.value) == 'A is already stopped'

#     with pytest.raises(Exception) as e:
#         assistant2.stop()
#         assistant2.stop()  # stop twice
#     assert str(e.value) == 'B is already stopped'

#     time.sleep(JOB_DELAY * 10) # allow cleanup to happen
#     expected = '''
# 12345678910
# 1001100210031004100510061007100810091010'''
#     assert doc.get() == expected


# def test_concurrent_assistants_stop_by_finishing():
#     '''
#     Test where threads allowed to finish, and never explicitly stopped
#     '''
#     executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

#     doc = ThreadSafeVar('')
#     app_fn = append_applicator_fn(doc)
#     edits = Edits(app_fn, job_delay=JOB_DELAY)

#     n1 = "A"
#     start_tag_1 = ':START_A:'
#     end_tag_1 = ':END_A:'
#     assistant1 = Assistant(
#         name=n1,
#         executor=executor,
#         edits=edits,
#         on_start_but_running=on_start_but_running(n1),
#         on_stop_but_stopped=on_stop_but_stopped(n1),
#     )

#     n2 = "B"
#     start_tag_2 = ':START_B:'
#     end_tag_2 = ':END_B:'
#     assistant2 = Assistant(
#         name=n2,
#         executor=executor,
#         edits=edits,
#         on_start_but_running=on_start_but_running(n2),
#         on_stop_but_stopped=on_stop_but_stopped(n2),
#     )

#     # Create job queues for each assistant
#     edits.create_job_queue(n1)
#     edits.create_job_queue(n2)

#     # Start the edits thread
#     edits.start()

#     ##########
#     # ROUND 1

#     # Start Assistant1
#     assistant1.start(init_function(n1, n1, 1, -1),
#                      streaming_function(n1, start_tag_1, end_tag_1, 1, 4),  # << shorter run
#                      cleanup_function(n1, start_tag_1, end_tag_1))

#     # Start Assistant2
#     assistant2.start(init_function(n2, n2, 2, -1),
#                      streaming_function(n2, start_tag_2, end_tag_2, 1001, 1004),  # << shorter run
#                      cleanup_function(n2, start_tag_2, end_tag_2))
#     assistant2.stop()  # test explicit stopping

#     # Let them run for 10x the delay streaming_function
#     time.sleep(JOB_DELAY*100)


#     ##########
#     # ROUND 2

#     # Start Assistant1
#     assistant1.start(init_function(n1, n1, -1, -1),
#                      streaming_function(n1, start_tag_1, end_tag_1, 1, 4),  # << shorter run
#                      cleanup_function(n1, start_tag_1, end_tag_1))

#     # Start Assistant2
#     assistant2.start(init_function(n2, n2, -1, -1),
#                      streaming_function(n2, start_tag_2, end_tag_2, 1001, 1004),  # << shorter run
#                      cleanup_function(n2, start_tag_2, end_tag_2))

#     # Let them run for 10x the delay streaming_function
#     time.sleep(JOB_DELAY*100)

#     expected = '''
# 123
# 1001
# 123
# 100110021003'''
#     assert doc.get() == expected
