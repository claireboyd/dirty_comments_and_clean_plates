export OPENAI_API_KEY=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
source /opt/homebrew/opt/chruby/share/chruby/chruby.sh
source /opt/homebrew/opt/chruby/share/chruby/auto.sh
chruby ruby-3.1.3
export PATH="/opt/homebrew/opt/openjdk@11/bin:$PATH"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/Users/claireboyd/anaconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/Users/claireboyd/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/Users/claireboyd/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/Users/claireboyd/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

