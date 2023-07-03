'''

FAILED

This is a failed attempt at using emacs markers. Keeping the state coordinated
across server and client was unmanageable.

The alternative, which is also relevant cross-editor, is to use text tags within the document.

'''

##################################################
# Marker
#

# Global variable to store the marker
emacs_markers = {}

@server.thread()
@server.command('command.markerUpdate')
def marker_update(ls: LanguageServer, args):
    global emacs_markers
    doc = args[0]
    params_list = args[1] # params_list = [{'line': 3, 'character': 0}, ...]
    print(f'DOC: {doc}')
    print(f'PARAMS_LIST: {params_list}')

    text_document = converter.structure(doc, TextDocumentIdentifier)
    uri = text_document.uri

    # Initialize the list of markers for this document if it doesn't exist
    if uri not in emacs_markers:
        emacs_markers[uri] = {}

    # Extract the EmacsMarkers
    for key, marker in  params_list.items():
        if marker:
            emacs_markers[uri][key] = marker  # {line: ..., character: ...}
            print(f'SET: Set marker {key} at line: {marker["line"]} and character: {marker["character"]}')
        else:
            print('No marker received.')

    return {'status': 'success'}


##################################################
# Emacs code:
#
# Each command would call:
#
#   (marker-set :api)

;;;;;;;;;;
;; MARKERS

(cl-defstruct marker-data
  "A structure for marker data. Contains the marker, its overlay, and associated colors."
  (marker (make-marker))
  (overlay nil)
  (face 'highlight))

(defvar-local markers-plist
  `(:transcription ,(make-marker-data :marker (point-marker) :face 'hi-yellow)
    :local ,(make-marker-data :marker (point-marker) :face 'hi-green)
    :api ,(make-marker-data :marker (point-marker) :face 'hi-blue))
  )

(defun get-marker-column (marker)
  "Get column number of a marker"
  (save-excursion
    (goto-char marker)
    (current-column)))

(defun marker-set (keyword)
  "Update an Emacs marker in the markers-plist."
  (interactive)
  (let ((marker-data (plist-get markers-plist keyword)))
    (when marker-data
      (setf (marker-data-marker marker-data) (point-marker))
      (plist-put markers-plist keyword marker-data)
      (marker-update-command markers-plist)
      )))

(defun convert-markers-plist (original-plist)
  "Converts original-plist into a new plist where each keyword is coupled to line number and character number"
  (let ((new-plist '())
        (current-plist original-plist))
    (while current-plist
      (let* ((key (pop current-plist))
             (value (pop current-plist))
             (marker (marker-data-marker value))
             (line-number (line-number-at-pos marker))
             (character-number (get-marker-column marker)))
        (setq new-plist (append new-plist
                                (list key
                                      (list :line line-number
                                            :character character-number))))))
    new-plist))

(defun marker-update-command (marker-data-plist)
  "Update markers and send marker positions to the LSP server."
  (interactive)
  ;; build jsonifyable plist of markers
  (let* ((doc (eglot--TextDocumentIdentifier))
         (params (convert-markers-plist marker-data-plist))
         (args (vector doc params)))
    (message "CALLING UPDATE: %s" args)
    (eglot-execute-command (eglot--current-server-or-lose) 'command.markerUpdate args))

  ;; Remove the old overlays, if any
  (mapc (lambda (marker-data)
          (let ((overlay (marker-data-overlay marker-data)))
            (when (overlayp overlay)
              (delete-overlay overlay))))
        ;; plist values
        (cl-loop for (_ v) on marker-data-plist by #'cddr collect v)
        )

  ;; Create new overlays at the marker's positions
  (mapc (lambda (marker-data)
          (let ((marker (marker-data-marker marker-data)))
            (when (marker-buffer marker)  ;; only when marker is not nil
              (let ((marker-pos (marker-position marker)))
                (setf (marker-data-overlay marker-data) (make-overlay marker-pos (1+ marker-pos)))
                (overlay-put (marker-data-overlay marker-data) 'face (marker-data-face marker-data))))))
        ;; plist values
        (cl-loop for (_ v) on marker-data-plist by #'cddr collect v)
        )
  )

(defun marker-update-hook (begin end length)
  "Call `marker-update-command' if the current buffer is managed by Eglot."
  (message "TRYING UPDATE HOOK")
  (when (bound-and-true-p eglot--managed-mode)
    (let ((markers-affected (cl-some (lambda (marker-data) (<= begin (marker-position (marker-data-marker marker-data))))
                                     (list (plist-get markers-plist :transcription)
                                           (plist-get markers-plist :local)
                                           (plist-get markers-plist :api)))))
      (when markers-affected
        (marker-update-command markers-plist)))))

(add-hook 'after-change-functions #'marker-update-hook)
