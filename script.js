document.addEventListener('DOMContentLoaded', () => {
    let wordChecks = {};

    // get DOM elements
    const contentEditable = document.querySelector('.input-field');
    const flaggedSection = document.querySelector('.overview-section:nth-child(2) .edit-section');
    const suggestionsSection = document.querySelector('.overview-section:nth-child(3) .edit-section');
    const countDisplay = document.querySelector('.count-display');
    const helpButton = document.querySelector('.help-button');
    const modalOverlay = document.querySelector('.modal-overlay');
    const modalClose = document.querySelector('.modal-close');
    const formatButtons = document.querySelectorAll('.format-btn');
    const clearButton = document.querySelector('.clear-btn');
    const rescanButton = document.querySelector('.rescan-btn');
    const copyButton = document.querySelector('.copy-btn');
    const wordCountDisplay = document.querySelector('.word-count');

    // fetch updated wordChecks from Flask server
    async function fetchUpdatedWordChecks() {
        try {
            const text = contentEditable.textContent;
            const response = await fetch('http://127.0.0.1:5000/update_word_checks', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text })
            });
            if (response.ok) {
                wordChecks = await response.json();
                console.log('Updated wordChecks:', wordChecks);
                updateContent();
            } else {
                console.error('Failed to fetch updated wordChecks');
            }
        } catch (error) {
            console.error('Error:', error);
        }
    }

    // format button
    formatButtons.forEach(button => {
        button.addEventListener('click', () => {
            const command = button.dataset.command;
            
            if (command === 'createLink') {
                const url = prompt('Enter the URL:');
                if (url) {
                    document.execCommand(command, false, url);
                }
            } else if (command === 'removeFormat') {
                document.execCommand(command, false, null);
                document.execCommand('unlink', false, null);
                // Remove any custom highlight spans
                const text = contentEditable.textContent;
                contentEditable.textContent = text;
            } else {
                document.execCommand(command, false, null);
            }
            
            // Toggle active state for formatting buttons
            if (['bold', 'italic'].includes(command)) {
                button.classList.toggle('active');
            }
            
            fetchUpdatedWordChecks();
        });
    });

    // clear and re-scan
    if (clearButton) {
        clearButton.addEventListener('click', () => {
            contentEditable.innerHTML = '';
            updateWordCount();
            fetchUpdatedWordChecks();
        });
    }

    if (rescanButton) {
        rescanButton.addEventListener('click', () => {
            fetchUpdatedWordChecks();
        });
    }

    function updateContent() {
        // save selection state
        const selection = window.getSelection();
        const range = selection.rangeCount > 0 ? selection.getRangeAt(0) : null;

        let html = contentEditable.innerHTML;
        let foundFlagged = new Set();
        let foundSuggestions = new Set();

        // process flagged words
        Object.entries(wordChecks.flagged || {}).forEach(([word, reason]) => {
            const regex = new RegExp(`\\b${word}\\b`, 'gi');
            if (regex.test(html)) {
                foundFlagged.add({ word, reason });
            }
            regex.lastIndex = 0;
            html = html.replace(regex, `<span class="flagged-word" data-word="${word}">$&</span>`);
        });

        // process suggested words
        Object.entries(wordChecks.suggestions || {}).forEach(([word, suggestion]) => {
            const regex = new RegExp(`\\b${word}\\b`, 'gi');
            if (regex.test(html)) {
                foundSuggestions.add({ word, suggestion });
            }
            regex.lastIndex = 0;
            html = html.replace(regex, `<span class="suggested-word" data-word="${word}">$&</span>`);
        });
        contentEditable.innerHTML = html;

        if (range) {
            try {
                selection.removeAllRanges();
                selection.addRange(range);
            } catch (e) {
                placeCaretAtEnd(contentEditable);
            }
        }

        updateFlaggedSection(foundFlagged);
        updateSuggestionsSection(foundSuggestions);
        countDisplay.textContent = foundFlagged.size + foundSuggestions.size;
    }

    function updateFlaggedSection(foundWords) {
        flaggedSection.innerHTML = '';
        foundWords.forEach(({ word, reason }) => {
            const flaggedItem = `
                <div class="edit-item">
                    <div class="edit-header">
                        <div class="flagged-dot"></div>
                        <div>${reason}</div>
                    </div>
                    <div class="word-correction">
                        <span>${word}</span>
                    </div>
                    <div class="button-group">
                        <button class="btn btn-accept" data-word="${word}" data-type="flagged">Delete</button>
                        <button class="btn btn-dismiss" data-type="flagged">Dismiss</button>
                    </div>
                </div>
            `;
            flaggedSection.insertAdjacentHTML('beforeend', flaggedItem);
        });
    }

    function updateSuggestionsSection(foundWords) {
        suggestionsSection.innerHTML = '';
        foundWords.forEach(({ word, suggestion }) => {
            const suggestionItem = `
                <div class="edit-item">
                    <div class="edit-header">
                        <div class="suggestion-dot"></div>
                        <div>Consider revising</div>
                    </div>
                    <div class="word-correction">
                        <span class="strikethrough">${word}</span>
                        <span>${suggestion}</span>
                    </div>
                    <div class="button-group">
                        <button class="btn btn-accept" data-word="${word}" data-type="suggestion">Accept</button>
                        <button class="btn btn-dismiss" data-type="suggestion">Dismiss</button>
                    </div>
                </div>
            `;
            suggestionsSection.insertAdjacentHTML('beforeend', suggestionItem);
        });
    }

    function placeCaretAtEnd(element) {
        const range = document.createRange();
        const selection = window.getSelection();
        range.selectNodeContents(element);
        range.collapse(false);
        selection.removeAllRanges();
        selection.addRange(range);
    }

    function deleteWordFromInput(word) {
        const text = contentEditable.innerHTML;
        const regexFlagged = new RegExp(`<span class="flagged-word" data-word="${word}">${word}</span>`, 'g');
        contentEditable.innerHTML = text.replace(regexFlagged, '');
        updateContent();
    }

    function replaceWordFromInput(word) {
        const text = contentEditable.innerHTML;
        const regexSuggested = new RegExp(`<span class="suggested-word" data-word="${word}">${word}</span>`, 'g');
        contentEditable.innerHTML = text.replace(regexSuggested, wordChecks.suggestions[word]);
        updateContent();
    }

    document.addEventListener('click', (e) => {
        if (e.target.matches('.btn-accept')) {
            const word = e.target.dataset.word;
            const type = e.target.dataset.type;
            if (type === 'flagged') {
                deleteWordFromInput(word);
            } else if (type === 'suggestion') {
                replaceWordFromInput(word);
            }
        } else if (e.target.matches('.btn-dismiss')) {
            const editItem = e.target.closest('.edit-item');
            if (editItem) {
                editItem.remove();
                const remainingItems = document.querySelectorAll('.edit-item').length;
                countDisplay.textContent = remainingItems;
            }
        }
    });

    // copy functionality
    copyButton.addEventListener('click', () => {
        const text = contentEditable.textContent;
        navigator.clipboard.writeText(text).then(() => {
            copyButton.innerHTML = '<i class="fa fa-check"></i>';
            setTimeout(() => {
                copyButton.innerHTML = '<i class="fa fa-copy"></i>';
            }, 2000);
        });
    });

    // word count functionality
    function updateWordCount() {
        const text = contentEditable.textContent || '';
        const wordCount = text.trim() === '' ? 0 : text.trim().split(/\s+/).length;
        wordCountDisplay.textContent = `${wordCount} words`;
    }

    helpButton.addEventListener('click', () => {
        modalOverlay.style.display = 'flex';
    });

    modalClose.addEventListener('click', () => {
        modalOverlay.style.display = 'none';
    });

    modalOverlay.addEventListener('click', (e) => {
        if (e.target === modalOverlay) {
            modalOverlay.style.display = 'none';
        }
    });

    contentEditable.addEventListener('focus', function() {
        if (this.textContent === 'Type or paste (Command + V) text here or upload a document') {
            this.textContent = '';
        }
    }, { once: true });

    // setting up event listeners
    contentEditable.addEventListener('input', () => {
        updateWordCount();
        fetchUpdatedWordChecks();
    });

    // setup
    fetchUpdatedWordChecks();
    updateWordCount();
});