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

(add-hook 'llm-mode-hook
          (lambda ()
            (define-key llm-mode-map (kbd "C-c l g") 'eglot-code-action-openai-gpt)
            (define-key llm-mode-map (kbd "C-c l c") 'eglot-code-action-openai-chatgpt)
            (define-key llm-mode-map (kbd "C-c l l") 'eglot-code-action-local-llm)
            (define-key llm-mode-map (kbd "C-c C-c") 'eglot-code-action-local-llm)
            (define-key llm-mode-map (kbd "C-c l s") 'eglot-code-action-stop-local-llm)
            (eglot-ensure)))

(require 'eglot)
(add-to-list 'eglot-server-programs
             `(llm-mode . ("localhost" 5033)))
