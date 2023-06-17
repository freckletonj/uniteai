# Add Emacs Marker for Transcription

The emacs marker can keep track of what point should be written to.

## An example of dealing with a marker

```elisp
;;;;;;;;;;
;; Marker Test

(defvar-local my-global-marker nil)

(defun my-initialize-marker ()
  "Set the global marker to the beginning of the buffer."
  (setq my-global-marker (point-min-marker)))

(add-hook 'find-file-hook 'my-initialize-marker)

(defvar-local my-marker-overlay nil
  "Overlay for the marker set by `marker-set-command'.")

(defun get-marker-column (marker)
  "Get column number of a marker"
  (save-excursion
    (goto-char marker)
    (current-column)))

(defun marker-update-command()
  (interactive)
  ;; Report marker
  (let* ((doc (eglot--TextDocumentIdentifier))
         (line (line-number-at-pos my-global-marker))
         (character (get-marker-column my-global-marker))
         (params `(:emacsMarker (:line ,line :character ,character))))
    (eglot-execute-command (eglot--current-server-or-lose) 'command.markerSet (vector doc params)))

  ;; Remove the old overlay, if any
  (when (overlayp my-marker-overlay)
    (delete-overlay my-marker-overlay))

  ;; Create a new overlay at the marker's position
  (let ((marker-pos (marker-position my-global-marker)))
    (setq my-marker-overlay (make-overlay marker-pos (1+ marker-pos)))
    (overlay-put my-marker-overlay 'face 'highlight))
  )

(defun marker-set-command ()
  "Send an Emacs marker to the LSP server."
  (interactive)
  (setq my-global-marker (point-marker))
  (marker-update-command))

(defun marker-get-command ()
  "Get the Emacs marker from the LSP server."
  (interactive)
  (let* ((doc (eglot--TextDocumentIdentifier))
         (marker my-global-marker)
         (line (line-number-at-pos marker))
         (character (current-column))
         (params `(:emacsMarker (:line ,line :character ,character))))
    (eglot-execute-command (eglot--current-server-or-lose) 'command.markerGet (vector doc params))))

(defun my-after-change-function (begin end length)
  "Call `marker-set-command' if the current buffer is managed by Eglot."
  (when (bound-and-true-p eglot--managed-mode)
    (marker-update-command)))

(add-hook 'after-change-functions #'my-after-change-function)
```
