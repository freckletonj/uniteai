;; ADD TO init.el

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; LLM Mode
;;
;; This is only useful at the current stage of development, and should be
;; removed once `lsp-mode` is integrated and allows concurrent LSPs per each
;; buffer.

(define-derived-mode llm-mode fundamental-mode "llm"
  "A mode for llm files."
  (setq-local comment-start "#")
  (setq-local comment-start-skip "#+\\s-*"))

(defvar llm-mode-map
  (let ((map (make-sparse-keymap)))
    map)
  "Keymap for `llm-mode'.")

(defvar llm-mode-hook nil)

(provide 'llm-mode)

(add-to-list 'auto-mode-alist '("\\.llm\\'" . llm-mode))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; LLM Mode (TODO REDUNDANT)

(add-to-list 'load-path (expand-file-name "~/_/llmpal/"))

(require 'llm-mode)

(use-package llm-mode
  :ensure nil
  :mode ("\\.llm\\'" . llm-mode)
  :hook (llm-mode . eglot-ensure))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; EGlot

(use-package eglot
  :ensure t
  :hook
  (eglot--managed-mode . company-mode)
  :init
  ;; Tell eglot not to ask if you're ok with the server modifying the document.
  (setq eglot-confirm-server-initiated-edits nil)
  :config
  (define-key eglot-mode-map (kbd "M-'") 'eglot-code-actions)
  )


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


;;;;;;;;;;

(defun eglot-code-action-openai-gpt ()
  (interactive)
  (unless mark-active
    (error "No region selected"))
  (marker-set :api)
  (let* ((server (eglot--current-server-or-lose))
         (doc (eglot--TextDocumentIdentifier))
         (range (list :start (eglot--pos-to-lsp-position (region-beginning))
                      :end (eglot--pos-to-lsp-position (region-end)))))
    (eglot-execute-command server 'command.openaiAutocompleteStream (vector doc range "FROM_CONFIG_COMPLETION" "FROM_CONFIG"))))

(defun eglot-code-action-openai-chatgpt ()
  (interactive)
  (unless mark-active
    (error "No region selected"))
  (marker-set :api)
  (let* ((server (eglot--current-server-or-lose))
         (doc (eglot--TextDocumentIdentifier))
         (range (list :start (eglot--pos-to-lsp-position (region-beginning))
                      :end (eglot--pos-to-lsp-position (region-end)))))
    (eglot-execute-command server 'command.openaiAutocompleteStream (vector doc range "FROM_CONFIG_CHAT" "FROM_CONFIG"))))

(defun eglot-code-action-local-llm ()
  (interactive)
  (unless mark-active
    (error "No region selected"))
  (marker-set :local)
  (let* ((server (eglot--current-server-or-lose))
         (doc (eglot--TextDocumentIdentifier))
         (range (list :start (eglot--pos-to-lsp-position (region-beginning))
                      :end (eglot--pos-to-lsp-position (region-end)))))
    (eglot-execute-command server 'command.localLlmStream (vector doc range))))

(defun eglot-code-action-stop-local-llm ()
  (interactive)
  (eglot-execute-command (eglot--current-server-or-lose) 'command.localLlmStreamStop '()))

(defun eglot-code-action-transcribe-stream ()
  (interactive)
  (marker-set :transcription)
  (let* ((server (eglot--current-server-or-lose))
         (doc (eglot--TextDocumentIdentifier)))
    (eglot-execute-command server 'command.transcribeStream (vector doc))))

(defun eglot-code-action-stop-transcribe-stream ()
  (interactive)
  (eglot-execute-command (eglot--current-server-or-lose) 'command.stopTranscribeStream '()))



;;;;;;;;;;

(add-hook 'llm-mode-hook
          (lambda ()
            (define-key llm-mode-map (kbd "C-c l g") 'eglot-code-action-openai-gpt)
            (define-key llm-mode-map (kbd "C-c l c") 'eglot-code-action-openai-chatgpt)
            (define-key llm-mode-map (kbd "C-c l l") 'eglot-code-action-local-llm)
            (define-key llm-mode-map (kbd "C-c C-c") 'eglot-code-action-local-llm)
            (define-key llm-mode-map (kbd "C-c l s") 'eglot-code-action-stop-local-llm)

            (define-key llm-mode-map (kbd "C-c l v") 'eglot-code-action-transcribe-stream)
            (define-key llm-mode-map (kbd "C-c l b") 'eglot-code-action-stop-transcribe-stream)

            (eglot-ensure)))

(require 'eglot)
(add-to-list 'eglot-server-programs
             `(llm-mode . ("localhost" 5033)))
