document.addEventListener('DOMContentLoaded', () => {
    let wordChecks = {};

    // Function to fetch updated wordChecks from Flask server
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
                updateContent(); // Call updateContent after fetching wordChecks
            } else {
                console.error('Failed to fetch updated wordChecks');
            }
        } catch (error) {
            console.error('Error:', error);
        }
    }

    // Get DOM elements
    const contentEditable = document.querySelector('.input-field');
    const flaggedSection = document.querySelector('.overview-section:nth-child(2) .edit-section');
    const suggestionsSection = document.querySelector('.overview-section:nth-child(3) .edit-section');
    const countDisplay = document.querySelector('.count-display');
    const helpButton = document.querySelector('.help-button');
    const modalOverlay = document.querySelector('.modal-overlay');
    const modalClose = document.querySelector('.modal-close');
    const formatButtons = document.querySelectorAll('.format-btn');
    const clearButton = document.querySelectorAll('.clear-btn');
    const rescanButton = document.querySelectorAll('.rescan-btn');
    const copyButton = document.querySelector('.copy-btn');
    const wordCountDisplay = document.querySelector('.word-count');
    
    
    // Function to highlight words and update panels
    function updateContent() {
        let text = contentEditable.textContent;
        let foundFlagged = new Set();
        let foundSuggestions = new Set();

        // Check flagged words
        Object.entries(wordChecks.flagged).forEach(([word, reason]) => {
            const regex = new RegExp(`\\b${word}\\b`, 'gi');
            if (regex.test(text)) {
                foundFlagged.add({ word, reason });
            }
            regex.lastIndex = 0;
            text = text.replace(regex, `<span class="flagged-word" data-word="${word}">$&</span>`);
        });

        // Check suggested improvements
        Object.entries(wordChecks.suggestions).forEach(([word, suggestion]) => {
            const regex = new RegExp(`\\b${word}\\b`, 'gi');
            if (regex.test(text)) {
                foundSuggestions.add({ word, suggestion });
            }
            regex.lastIndex = 0;
            text = text.replace(regex, `<span class="suggested-word" data-word="${word}">$&</span>`);
        });

        // Update the content
        contentEditable.innerHTML = text;
        placeCaretAtEnd(contentEditable);
        
        // Update panels
        updateFlaggedSection(foundFlagged);
        updateSuggestionsSection(foundSuggestions);
        
        // Update count
        countDisplay.textContent = foundFlagged.size + foundSuggestions.size;
    }

    // Function to update the flagged section
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

    // Function to update the suggestions section
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

    // Helper function to place cursor at end
    function placeCaretAtEnd(element) {
        const range = document.createRange();
        const selection = window.getSelection();
        range.selectNodeContents(element);
        range.collapse(false);
        selection.removeAllRanges();
        selection.addRange(range);
    }

    // Function to delete a word from input
    function deleteWordFromInput(word) {
        const text = contentEditable.innerHTML;
        const regexFlagged = new RegExp(`<span class="flagged-word" data-word="${word}">${word}</span>`, 'g');
        contentEditable.innerHTML = text.replace(regexFlagged, '');
        updateContent();
    }

    //Function to switch word from input 
    function replaceWordFromInput(word) {
        const text = contentEditable.innerHTML;
        const regexSuggested = new RegExp(`<span class="suggested-word" data-word="${word}">${word}</span>`, 'g');
        contentEditable.innerHTML = text.replace(regexSuggested, wordChecks.suggestions[word]);
        updateContent();
    }
    
    // Event delegation for button clicks
    document.addEventListener('click', (e) => {
        if (e.target.matches('.btn-accept')) {
            const word = e.target.dataset.word;
            deleteWordFromInput(word);
            replaceWordFromInput(word);
        } else if (e.target.matches('.btn-dismiss')) {
            const editItem = e.target.closest('.edit-item');
            if (editItem) {
                editItem.remove();
                // Update count after dismissal
                const remainingItems = document.querySelectorAll('.edit-item').length;
                countDisplay.textContent = remainingItems;
            }
        }
    });

    // Modal functionality
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

    // THIS IS CURRENTLY BEING CALLED ON EDIT. NEED TO MAKE RUN BUTTON
    contentEditable.addEventListener('input', fetchUpdatedWordChecks);
    fetchUpdatedWordChecks();

    // Copy functionality
    copyButton.addEventListener('click', () => {
        const text = contentEditable.textContent;
        navigator.clipboard.writeText(text).then(() => {
            // Visual feedback
            copyButton.innerHTML = '<i class="fa fa-check"></i>';
            setTimeout(() => {
                copyButton.innerHTML = '<i class="fa fa-copy"></i>';
            }, 2000);
        });
    });

    // Word count functionality
    function updateWordCount() {
        const text = contentEditable.textContent || '';
        const wordCount = text.trim() === '' ? 0 : text.trim().split(/\s+/).length;
        wordCountDisplay.textContent = `${wordCount} words`;
    }

    // Add word count update to existing input listener
    contentEditable.addEventListener('input', () => {
        updateContent(); // Your existing update function
        updateWordCount();
    });

    // Initial word count
    updateWordCount();

    function clearText(element) {
        if (element.innerHTML === 'Type or paste (Command + V) text here or upload a document') {
            element.innerHTML = '';
        }
    }

    function restoreText(element) {
        if (element.innerHTML === '') {
            element.innerHTML = 'Type or paste (Command + V) text here or upload a document';
        }
    }
    
    clearButton.addEventListener('click', () => {
        clearText();
    });
    rescanButton.addEventListener('click', () => {
        restoreText();
    });


});

