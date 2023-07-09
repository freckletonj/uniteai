;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

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

;; Global stopping
(defun eglot-stop ()
  (interactive)
  (let* ((server (eglot--current-server-or-lose))
         (doc (eglot--TextDocumentIdentifier)))
    (eglot-execute-command server 'command.stop (vector doc))))

;; Example Counter
(defun eglot-example-counter ()
  (interactive)
  (let* ((server (eglot--current-server-or-lose))
         (doc (eglot--TextDocumentIdentifier))
         (pos (eglot--pos-to-lsp-position (point))))
    (eglot-execute-command server 'command.exampleCounter (vector doc pos))))

;; Local LLM
(defun eglot-local-llm ()
  (interactive)
  (unless mark-active
    (error "No region selected"))
  (let* ((server (eglot--current-server-or-lose))
         (doc (eglot--TextDocumentIdentifier))
         (range (list :start (eglot--pos-to-lsp-position (region-beginning))
                      :end (eglot--pos-to-lsp-position (region-end)))))
    (eglot-execute-command server 'command.localLlmStream (vector doc range))))

;; Transcription
(defun eglot-transcribe ()
  (interactive)
  (let* ((server (eglot--current-server-or-lose))
         (doc (eglot--TextDocumentIdentifier))
         (pos (eglot--pos-to-lsp-position (point))))
    (eglot-execute-command server 'command.transcribe (vector doc pos))))

;; OpenAI
(defun eglot-openai-gpt ()
  (interactive)
  (unless mark-active
    (error "No region selected"))
  (let* ((server (eglot--current-server-or-lose))
         (doc (eglot--TextDocumentIdentifier))
         (range (list :start (eglot--pos-to-lsp-position (region-beginning))
                      :end (eglot--pos-to-lsp-position (region-end)))))
    (eglot-execute-command server 'command.openaiAutocompleteStream (vector doc range "FROM_CONFIG_COMPLETION" "FROM_CONFIG"))))

(defun eglot-openai-chatgpt ()
  (interactive)
  (unless mark-active
    (error "No region selected"))
  (let* ((server (eglot--current-server-or-lose))
         (doc (eglot--TextDocumentIdentifier))
         (range (list :start (eglot--pos-to-lsp-position (region-beginning))
                      :end (eglot--pos-to-lsp-position (region-end)))))
    (eglot-execute-command server 'command.openaiAutocompleteStream (vector doc range "FROM_CONFIG_CHAT" "FROM_CONFIG"))))

(add-hook 'llm-mode-hook
          (lambda ()
            (define-key llm-mode-map (kbd "C-c l s") 'eglot-stop)
            (define-key llm-mode-map (kbd "C-c l e") 'eglot-example-counter)
            (define-key llm-mode-map (kbd "C-c l l") 'eglot-local-llm)
            (define-key llm-mode-map (kbd "C-c C-c") 'eglot-local-llm)
            (define-key llm-mode-map (kbd "C-c l v") 'eglot-transcribe)
            (define-key llm-mode-map (kbd "C-c l g") 'eglot-openai-gpt)
            (define-key llm-mode-map (kbd "C-c l c") 'eglot-openai-chatgpt)
            (eglot-ensure)))

(require 'eglot)
(add-to-list 'eglot-server-programs
             `(llm-mode . ("localhost" 5033)))
