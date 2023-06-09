;;; llm-mode.el --- major mode for editing LLM files

;; Copyright (C) 2023 Your Name

;; Author: Your Name
;; Version: 0.1
;; Package-Requires: ((emacs "24.1") (eglot "1.6"))
;; Keywords: languages

;;; Commentary:

;; This is a major mode for editing LLM files.

;;; Code:

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

;;; llm-mode.el ends here



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;



;; ;; Define llm-mode and associate it with the .llm file extension
;; (define-generic-mode 'llm-mode
;;   '("#")                                        ; comment character
;;   '()                                           ; keywords
;;   '()                                           ; other highlighting
;;   '("\\.llm$")                                  ; file extensions
;;   nil                                        ; function to run
;;   "A mode for llm files"                        ; doc string
;; )


;; ;; Define llm-mode and associate it with the .llm file extension
;; (define-derived-mode llm-mode fundamental-mode "llm"
;;   "A mode for llm files."
;;   (setq-local comment-start "#")
;;   (setq-local comment-start-skip "#+\\s-*")
;;   )

;; ;; Create a keymap for llm-mode
;; (defvar llm-mode-map
;;   (let ((map (make-sparse-keymap)))
;;     ;; Define keybindings here
;;     ;; For example, we can bind the key combination "C-c C-a" to the function 'example-function
;;     map)
;;   "Keymap for `llm-mode'.")

;; ;; Define a hook for llm-mode
;; (defvar llm-mode-hook nil)

;; (provide 'llm-mode)

;; (add-to-list 'auto-mode-alist '("\\.llm\\'" . llm-mode))

;; (require 'eglot)

;; (setq openai-max-length 300)

;; (defun eglot-code-action-openai-gpt ()
;;   (interactive)
;;   (let* ((server (eglot--current-server-or-lose))
;;          (doc (eglot--TextDocumentIdentifier))
;;          (range (list :start (eglot--pos-to-lsp-position (point-min))
;;                       :end (eglot--pos-to-lsp-position (point-max)))))
;;     (eglot-execute-command server 'command.openaiAutocompleteStream (vector doc range "text-davinci-002" openai-max-length))))

;; (defun eglot-code-action-openai-chatgpt ()
;;   (interactive)
;;   (let* ((server (eglot--current-server-or-lose))
;;          (doc (eglot--TextDocumentIdentifier))
;;          (range (list :start (eglot--pos-to-lsp-position (point-min))
;;                       :end (eglot--pos-to-lsp-position (point-max)))))
;;     (eglot-execute-command server 'command.openaiAutocompleteStream (vector doc range "gpt-3.5-turbo" openai-max-length))))

;; (defun eglot-code-action-local-llm ()
;;   (interactive)
;;   (let* ((server (eglot--current-server-or-lose))
;;          (doc (eglot--TextDocumentIdentifier))
;;          (range (list :start (eglot--pos-to-lsp-position (point-min))
;;                       :end (eglot--pos-to-lsp-position (point-max)))))
;;     (eglot-execute-command server 'command.localLlmStream (vector doc range))))

;; (defun eglot-code-action-stop-local-llm ()
;;   (interactive)
;;   (eglot-execute-command (eglot--current-server-or-lose) 'command.localLlmStreamStop '()))

;; (add-hook 'llm-mode-hook
;;           (lambda ()
;;             (define-key llm-mode-map (kbd "C-c l g") 'eglot-code-action-openai-gpt)
;;             (define-key llm-mode-map (kbd "C-c l c") 'eglot-code-action-openai-chatgpt)
;;             (define-key llm-mode-map (kbd "C-c l l") 'eglot-code-action-local-llm)
;;             (define-key llm-mode-map (kbd "C-c C-c") 'eglot-code-action-local-llm)
;;             (define-key llm-mode-map (kbd "C-c l s") 'eglot-code-action-stop-local-llm)
;;             ))

;; (add-to-list 'eglot-server-programs
;;              `(llm-mode . ("localhost" 5033))
;;              )
