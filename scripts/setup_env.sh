#!/bin/bash
# Setup environment variables for NFL Analytics
# This script helps configure .env from .env.example

set -e

ENV_FILE=".env"
EXAMPLE_FILE=".env.example"

echo "🔧 NFL Analytics Environment Setup"
echo "=================================="
echo ""

# Check if .env.example exists
if [ ! -f "$EXAMPLE_FILE" ]; then
    echo "❌ Error: $EXAMPLE_FILE not found"
    exit 1
fi

# Check if .env already exists
if [ -f "$ENV_FILE" ]; then
    echo "⚠️  $ENV_FILE already exists"
    read -p "Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "✅ Keeping existing $ENV_FILE"
        echo "   To add missing variables, run: diff .env .env.example"
        exit 0
    fi
fi

# Copy example to .env
cp "$EXAMPLE_FILE" "$ENV_FILE"
echo "✅ Created $ENV_FILE from $EXAMPLE_FILE"
echo ""

# Prompt for database password
echo "🔐 Database Configuration"
echo "------------------------"
read -sp "Enter DB_PASSWORD (required): " db_password
echo
if [ -z "$db_password" ]; then
    echo "ERROR: DB_PASSWORD is required"
    exit 1
fi

# Update .env with password
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s/DB_PASSWORD=CHANGE_ME/DB_PASSWORD=$db_password/" "$ENV_FILE"
else
    # Linux
    sed -i "s/DB_PASSWORD=CHANGE_ME/DB_PASSWORD=$db_password/" "$ENV_FILE"
fi

echo "✅ Updated DB_PASSWORD in $ENV_FILE"
echo ""

# Summary
echo "📋 Summary"
echo "----------"
echo "✅ Environment file created: $ENV_FILE"
echo "⚠️  IMPORTANT: Never commit $ENV_FILE to version control!"
echo ""
echo "Next steps:"
echo "  1. Review $ENV_FILE and update any other CHANGE_ME values"
echo "  2. Test database connection:"
echo "     psql postgresql://dro:$db_password@localhost:5544/devdb01"
echo ""

