#!/usr/bin/env bash
#
# Add MCMC convergence checks to R training scripts
# Usage: ./scripts/add_convergence_checks.sh [file.R]
#

set -e

add_convergence_check() {
    local file="$1"

    # Check if convergence checks already present
    if grep -q "max_rhat" "$file"; then
        echo "✅ $file: Convergence checks already present"
        return 0
    fi

    # Find line with model fitting
    if ! grep -q "fit.*<-.*brm(" "$file"; then
        echo "⚠️  $file: No brm() call found"
        return 0
    fi

    echo "Adding convergence checks to $file..."

    # Find the closing parenthesis of brm() call and insert after it
    # This is a simplified approach - for complex multiline calls, manual review needed
    echo "⚠️  Note: Convergence checks should be added AFTER the model fit completes"
    echo "    Please manually add the following after your brm() call:"
    echo ""
    echo "# MANDATORY: Convergence diagnostics"
    echo "max_rhat <- max(rhat(fit), na.rm = TRUE)"
    echo "min_ess <- min(neff_ratio(fit), na.rm = TRUE)"
    echo ""
    echo "if (max_rhat > 1.01) {"
    echo "  stop(\"❌ CONVERGENCE FAILURE: Rhat > 1.01\")"
    echo "}"
    echo ""
    echo "if (min_ess < 0.1) {  # ESS ratio"
    echo "  stop(\"❌ CONVERGENCE FAILURE: ESS ratio < 0.1\")"
    echo "}"
    echo ""
    echo "cat(\"✅ CONVERGENCE SUCCESS\\n\")"
    echo ""

    return 0

    echo "✅ Added convergence checks to $file"
}

# Main
if [ $# -eq 0 ]; then
    # Process all training scripts
    for file in R/train_*.R; do
        if [ -f "$file" ]; then
            add_convergence_check "$file"
        fi
    done
else
    # Process specific file
    add_convergence_check "$1"
fi
