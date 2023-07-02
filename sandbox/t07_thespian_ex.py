'''Playing with Thespian

Note:

  * Importantly, an Actor cannot access internals of other Actors, and it
    should not access any globals.

API: https://thespianpy.com/doc/using

  * tell - sends a message to a target Actor (as specified by the target
    Actor's ActorAddress)

  * ask - sends a message to a target Actor and waits for the Actor to send a
    response message. NOTE: does NOT get back a response to this ask, just gets
    any `send`ed response.

  * listen

  * send -

  * createActor - creates a new Actor. There is no Parent Actor for the newly
    created Actor; this Actor is referred to as a "top-level Actor" and it is
    managed by the ActorSystem itself. No other Actor will be notified if the
    top-level Actor exits.

  * .myAddress


  * `with ActorSystem().private(): ...` : route requests to correct thread, if
    using external application.


Actor Implementations
  https://thespianpy.com/doc/using#hH-2a5fa63d-e6eb-43b9-bea8-47223b27544e

'''

from thespian.actors import *
from dataclasses import dataclass
import time


@dataclass
class Greeting:
    message: str

class Hello(Actor):
    def __init__(self):
        self.i = 0

    def receiveMessage(self, message, sender):
        if message == 'hi':
            greeting = Greeting(f'Hello{self.i}')
            world = self.createActor(World)
            punct = self.createActor(Punctuate)
            greeting.sendTo = [punct, sender]  # send to punct, which then reads off sender, to send back to sender
            self.send(world, greeting)
            self.i += 1


class World(Actor):
    def __init__(self):
        self.i = 0

    def receiveMessage(self, message, sender):
        if isinstance(message, Greeting):
            message.message = message.message + f", World{self.i}"
            nextTo = message.sendTo.pop(0)
            self.send(nextTo, message)
            self.i += 1

class Punctuate(Actor):
    def __init__(self):
        self.i = 0

    def receiveMessage(self, message, sender):
        if isinstance(message, Greeting):
            nextTo = message.sendTo.pop(0)

            # NOTE: MUTABILITY SURPRISE!!!
            mx = message
            mx.message = mx.message + f"!{self.i}"
            self.send(nextTo, mx)

            # NOTE: `send` does NOT send to the original `ask`
            mx = message
            mx.message = mx.message + f"?{self.i}"
            self.send(nextTo, mx)
            self.i += 1


if __name__ == "__main__":
    a = ActorSystem()
    hello = a.createActor(Hello)
    print(a.ask(hello, 'hi', 0.2))
    print(a.ask(hello, 'hi', 0.2))
    time.sleep(0.5)
    a.tell(hello, ActorExitRequest())
    print(ActorSystem().ask(hello, 'hi', 0.2))
