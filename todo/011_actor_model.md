# 011 - Implement Actor Model

- Generic Assistant
  - init_block: writes tags
  - safe_to_edit: checks for the block's existence
  - write_to_block: send edit to queue
  - remove_tags:

- Transcription
  - start separate threads: 1) listen to mic, 2) transcribe it
  - in main transcription thread, await the transcription queue and send write out edits

- Local LLM
  - start streaming request
  - write out edits

- API LLM
  - start streaming request
  - write out edits
