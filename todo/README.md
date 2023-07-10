# Todo-directory's Readme

Kanbany tickets as part of the repo!

This section of the repo gets to be quite unstructured, so, sling some of your thoughts in here, make a PR against it, and get some feedback from the community!


## Dear Community:

- **Have expertise, but no time?** Please make a text-only PR against the markdown TODOs to help the next person get their bearings on a ticket.

- **Have bandwidth?** Feel free to grab a "ticket" and make a PR.

- **Have an idea?** Feel free to add a "ticket" and a PR on it, and the community can weigh in.


## Tehnical Details of Contributing

### Architecture

#### A High-Level `example`, [`./uniteai/contrib/example.py`](./uniteai/contrib/example.py)

Here's a feature that will start counting up inside your document, and just demonstrates how to build a new feature.


##### [`./.uniteai.yml`](./.uniteai.yml)

You specify what features should be included by adding their module path. For instance the following only allows the `example` feature:

```yaml
modules:
  # - uniteai.local_llm
  # - uniteai.transcription
  # - uniteai.openai
  - uniteai.contrib.example
```

The one central `.uniteai.yml` holds all config that may apply to any feature.

```yaml
# Example: `uniteai.contrib.example`
#
# a feature that counts up

start_digit: 13
end_digit: 42
delay: 0.4
```

##### [`./lsp_server.py`](./lsp_server.py) > `load_module`

Only desired modules will be loaded. They will be loaded via 2 functions they must provide:

```python
def configure(config_yaml):
    parser = argparse.ArgumentParser()
    ...
    parser.add_argument('--start_digit', default=config_yaml.get('start_digit', None))
    args, _ = parser.parse_known_args()
    return args


def initialize(config, server):
    # add an new actor to the `server`'s state
    server.add_actor(NAME, ExampleActor)

    # add commands to the `pygls` `Server`
    @server.thread()
    @server.command('command.exampleCounter')
    def example_counter(ls: Server, args):
        ...
        ls.tell_actor(NAME, actor_args)
```


##### [`./uniteai/contrib/example.py`](./uniteai/contrib/example.py)

This code stands for itself, more or less.

It includes:

* an `Actor` that can be communicated with via message passing.
* some code that streams edits via the `ls.edits` task runner
* `def configure(...)` and `def initialize(...)`, needed by `load_module` to surgically include/exclude desired/undesired features at runtime.

This is *not crazy ergonomic* yet, because I don't know how people will use this library, and I don't want to prematurely abstract some class that gets in the way of future development.

So, instead of introducing some leaky abstraction, some internals remain laid bare.

Please give me feedback of your experience as you learn from this example, and we'll slowly tighten things up. I envision some dangerous way of writing features, alongside some "happy-path" feature class that abstracts away most of this and merely requires a user provide a generator function like:

```python
def streaming_fn(highlighted_region_from_editor):
    for _ in whatever:
        yield intermediate_results
```


#### Actor Model

The internal pieces of this software communicate via the [Actor Model](https://en.wikipedia.org/wiki/Actor_model), using the [Thespian](https://github.com/thespianpy/Thespian) library.

Namely, they each run as there own separate threads, and they keep their own separate internal state. The only way to interact with an `Actor` is via mesage passing. For instance you can ask the `Transcription` actor to start transcribing, and if you ask twice in a row without `stop`ping in between, it has to determine how it handles that.

#### Edits Handler

Actors do not directly edit the document. Instead, edit tasks are passed off to single edits thread which handles actually applying them.

For instance, a `BlockJob` can be sent off to the `Edits` queue to add some text in the block which this actor owns, demarcated by the start and end tags:

```python
    job = BlockJob(
        uri=uri,
        start_tag=START_TAG,
        end_tag=END_TAG,
        text=f'\n{running_transcription}\n',
        strict=False,
    )
    edits.add_job(NAME, job)
```

Note this job is not `strict`. This means that it's ok if the edit applier thread fails to actually make this change. This is useful when streaming a response. Say 10 streamed edits have accumulated in the job queue. It's ok to skip the oldest 9 of them.


Right now `edit.py` accepts 3 classes of edits. This developer API should likely expand.

```

BlockJob: completely replace text within a block, demarcated by start and end tags.

DeleteJob: delete any region of the document (eg, clean up the start + end tags).

InsertJob: insert text anywhere (eg, place start + end tags).

```
