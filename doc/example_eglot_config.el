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
  (setq-local comment-start-skip "#+\\s-*")
  )

(defvar llm-mode-map
  (let ((map (make-sparse-keymap)))
    map)
  "Keymap for `llm-mode'.")

(defvar llm-mode-hook nil)

(provide 'llm-mode)

(add-to-list 'auto-mode-alist '("\\.llm\\'" . llm-mode))


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
  (define-key eglot-mode-map (kbd "M-'") 'eglot-code-actions))


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


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; LLM Mode

(add-to-list 'load-path (expand-file-name "~/_/llmpal/"))

(require 'llm-mode)

(use-package llm-mode
  :ensure nil
  :mode ("\\.llm\\'" . llm-mode)
  :hook (llm-mode . eglot-ensure))

(defun eglot-code-action-openai-gpt ()
  (interactive)
  (unless mark-active
    (error "No region selected"))
  (let* ((server (eglot--current-server-or-lose))
         (doc (eglot--TextDocumentIdentifier))
         (range (list :start (eglot--pos-to-lsp-position (region-beginning))
                      :end (eglot--pos-to-lsp-position (region-end)))))
    (eglot-execute-command server 'command.openaiAutocompleteStream (vector doc range "FROM_CONFIG_COMPLETION" "FROM_CONFIG"))))

(defun eglot-code-action-openai-chatgpt ()
  (interactive)
  (unless mark-active
    (error "No region selected"))
  (let* ((server (eglot--current-server-or-lose))
         (doc (eglot--TextDocumentIdentifier))
         (range (list :start (eglot--pos-to-lsp-position (region-beginning))
                      :end (eglot--pos-to-lsp-position (region-end)))))
    (eglot-execute-command server 'command.openaiAutocompleteStream (vector doc range "FROM_CONFIG_CHAT" "FROM_CONFIG"))))

(defun eglot-code-action-local-llm ()
  (interactive)
  (unless mark-active
    (error "No region selected"))
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
  (let* ((server (eglot--current-server-or-lose))
         (doc (eglot--TextDocumentIdentifier)))
    (eglot-execute-command server 'command.transcribeStream (vector doc))))

(defun eglot-code-action-stop-transcribe-stream ()
  (interactive)
  (eglot-execute-command (eglot--current-server-or-lose) 'command.stopTranscribeStream '()))



(defun my-call-sample-command ()
  "Send an Emacs marker to the LSP server."
  (interactive)
  (let* ((marker (point-marker))
         (line (line-number-at-pos marker))
         (character (current-column))
         (params `(:emacsMarker (:line ,line :character ,character))))
    (jsonrpc-async-request (eglot--current-server-or-lose)
                           :sampleCommand
                           params
                           :success-fn (lambda (result)
                                         (message "Command succeeded with result: %s" result))
                           :timeout-fn (lambda (result)
                                         (message "Command timed out")))))





(add-hook 'llm-mode-hook
          (lambda ()
            (define-key llm-mode-map (kbd "C-c l g") 'eglot-code-action-openai-gpt)
            (define-key llm-mode-map (kbd "C-c l c") 'eglot-code-action-openai-chatgpt)
            (define-key llm-mode-map (kbd "C-c l l") 'eglot-code-action-local-llm)
            (define-key llm-mode-map (kbd "C-c C-c") 'eglot-code-action-local-llm)
            (define-key llm-mode-map (kbd "C-c l s") 'eglot-code-action-stop-local-llm)

            (define-key llm-mode-map (kbd "C-c l v") 'eglot-code-action-transcribe-stream)
            (define-key llm-mode-map (kbd "C-c l b") 'eglot-code-action-stop-transcribe-stream)

            (define-key llm-mode-map (kbd "C-c l t") 'my-call-sample-command)
            (eglot-ensure)))

(require 'eglot)
(add-to-list 'eglot-server-programs
             `(llm-mode . ("localhost" 5033)))
