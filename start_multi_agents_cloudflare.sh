#!/bin/bash
# Start 1 Green Agent + 6 White Agents and expose to public internet
# Green Agent: 1 instance (port 8010)
# White Agents: 6 instances (ports 8011-8016)
#   - Agents 1-3: Web RAG mode
#   - Agents 4-6: Direct LLM mode (no RAG)

set -e

PROJECT_ROOT="$PWD"
GREEN_PORT=8010

# White agents configuration
# Format: DIRECTORY|PORT|MODEL|USE_RAG
# IMPORTANT: Each agent runs from its own directory to avoid .ab conflicts
declare -a WHITE_CONFIGS=(
    "white_agent_1|8011|mistralai/mistral-7b-instruct|true"
    "white_agent_2|8012|deepseek/deepseek-v3.2-exp|true"
    "white_agent_3|8013|openai/gpt-5.1|true"
    "white_agent_4|8014|mistralai/mistral-7b-instruct|false"
    "white_agent_5|8015|deepseek/deepseek-v3.2-exp|false"
    "white_agent_6|8016|openai/gpt-5.1|false"
)

echo "🚀 Starting Multi-Agent System + Cloudflare"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Configuration:"
echo "  Green Agent: 1 instance (port $GREEN_PORT)"
echo "  White Agents: ${#WHITE_CONFIGS[@]} instances (ports 8011-8016)"
echo ""
echo "  RAG Agents (1-3):"
for i in 0 1 2; do
    IFS='|' read -r dir port model use_rag <<< "${WHITE_CONFIGS[$i]}"
    echo "    #$((i+1)). $dir - Port $port - $model (Web RAG)"
done
echo ""
echo "  Direct LLM Agents (4-6):"
for i in 3 4 5; do
    IFS='|' read -r dir port model use_rag <<< "${WHITE_CONFIGS[$i]}"
    echo "    #$((i+1)). $dir - Port $port - $model (Direct LLM, no RAG)"
done
echo ""

# Check cloudflared
if ! command -v cloudflared &> /dev/null; then
    echo "❌ cloudflared not installed"
    echo "Install: wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb && sudo dpkg -i cloudflared-linux-amd64.deb"
    exit 1
fi

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Stopping all services..."

    # Stop AgentBeats
    pkill -P $$ agentbeats 2>/dev/null || true

    # Stop Cloudflare tunnels
    pkill -P $$ cloudflared 2>/dev/null || true

    echo "✅ All services stopped"
    exit 0
}

trap cleanup INT TERM

# ============================================
# 1. Start Cloudflare Tunnels
# ============================================
echo "📡 Starting Cloudflare tunnels..."

# Green agent tunnel
cloudflared tunnel --url http://localhost:$GREEN_PORT > /tmp/cf_green.log 2>&1 &
CF_GREEN_PID=$!

# White agents tunnels
for config in "${WHITE_CONFIGS[@]}"; do
    IFS='|' read -r dir port model use_rag <<< "$config"
    cloudflared tunnel --url http://localhost:$port > /tmp/cf_white_$port.log 2>&1 &
    eval "CF_WHITE_${port}_PID=$!"
done

echo "Waiting for Cloudflare tunnels to start..."
sleep 12

# Extract URLs
echo "📍 Extracting public URLs..."

GREEN_URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /tmp/cf_green.log | head -1)
if [ -z "$GREEN_URL" ]; then
    echo "❌ Failed to get Green Agent URL"
    cat /tmp/cf_green.log
    cleanup
fi

GREEN_DOMAIN=$(echo $GREEN_URL | sed 's/https:\/\///')
echo "✅ Green Agent: $GREEN_URL"

# Extract all white agent URLs
declare -A WHITE_URLS
declare -A WHITE_DOMAINS
declare -A WHITE_MODELS
declare -A WHITE_USE_RAGS
declare -A WHITE_DIRS

for config in "${WHITE_CONFIGS[@]}"; do
    IFS='|' read -r dir port model use_rag <<< "$config"

    url=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' /tmp/cf_white_$port.log | head -1)
    if [ -z "$url" ]; then
        echo "❌ Failed to get White Agent (port $port) URL"
        cat /tmp/cf_white_$port.log
        cleanup
    fi

    WHITE_URLS[$port]=$url
    WHITE_DOMAINS[$port]=$(echo $url | sed 's/https:\/\///')
    WHITE_MODELS[$port]=$model
    WHITE_USE_RAGS[$port]=$use_rag
    WHITE_DIRS[$port]=$dir

    mode=$([[ "$use_rag" == "true" ]] && echo "Web RAG" || echo "Direct LLM")
    echo "✅ White Agent ($port - $model - $mode): $url"
done

echo ""

# ============================================
# 2. Start Green AgentBeats
# ============================================
echo "🟢 Starting Green AgentBeats..."

cd $PROJECT_ROOT

export CLOUDRUN_HOST=$GREEN_DOMAIN
export HTTPS_ENABLED=true
export AGENT_TYPE=green
export PORT=$GREEN_PORT

agentbeats run_ctrl > /tmp/green_agent.log 2>&1 &
GREEN_AB_PID=$!

echo "Waiting for Green Agent to start..."
sleep 5

if ! ps -p $GREEN_AB_PID > /dev/null; then
    echo "❌ Green AgentBeats failed to start"
    cat /tmp/green_agent.log
    cleanup
fi

echo "✅ Green AgentBeats running (PID: $GREEN_AB_PID)"

# ============================================
# 3. Start White AgentBeats (from their own directories)
# ============================================
echo ""
echo "⚪ Starting White AgentBeats..."

declare -A WHITE_AB_PIDS

for config in "${WHITE_CONFIGS[@]}"; do
    IFS='|' read -r dir port model use_rag <<< "$config"
    domain="${WHITE_DOMAINS[$port]}"
    mode=$([[ "$use_rag" == "true" ]] && echo "Web RAG" || echo "Direct LLM")

    echo "  Starting White Agent - $dir - Port $port - $model - $mode..."

    # Change to white agent's directory (important for separate .ab/)
    cd "$PROJECT_ROOT/$dir"

    # Export environment variables
    export CLOUDRUN_HOST=$domain
    export HTTPS_ENABLED=true
    export AGENT_TYPE=white
    export PORT=$port
    export WHITE_AGENT_MODEL=$model
    export USE_DIRECT_LLM=$([[ "$use_rag" == "true" ]] && echo "false" || echo "true")

    # Start AgentBeats from this directory
    agentbeats run_ctrl > /tmp/white_agent_$port.log 2>&1 &
    pid=$!
    WHITE_AB_PIDS[$port]=$pid

    # Short wait
    sleep 3

    if ! ps -p $pid > /dev/null; then
        echo "  ❌ White AgentBeats (port $port) failed to start"
        cat /tmp/white_agent_$port.log
    else
        echo "  ✅ White AgentBeats (port $port) running (PID: $pid)"
    fi

    # Return to project root
    cd "$PROJECT_ROOT"
done

echo ""
echo "Waiting for all Agents to fully start..."
sleep 10

# ============================================
# 4. Get Agent URLs
# ============================================
echo "📋 Getting Agent URLs..."

# Green Agent
GREEN_AGENT_URL=$(grep -oP 'Agent.*?/to_agent/[a-f0-9]+' /tmp/green_agent.log | tail -1 | grep -oP '/to_agent/[a-f0-9]+' || echo "")
if [ -n "$GREEN_AGENT_URL" ]; then
    GREEN_FULL_URL="$GREEN_URL$GREEN_AGENT_URL"
else
    echo "⚠️  Could not extract Green Agent ID from logs, please check manually"
    GREEN_FULL_URL="$GREEN_URL/to_agent/<check_logs>"
fi

# White Agents
declare -A WHITE_FULL_URLS
for config in "${WHITE_CONFIGS[@]}"; do
    IFS='|' read -r dir port model use_rag <<< "$config"

    white_agent_url=$(grep -oP 'Agent.*?/to_agent/[a-f0-9]+' /tmp/white_agent_$port.log | tail -1 | grep -oP '/to_agent/[a-f0-9]+' || echo "")
    if [ -n "$white_agent_url" ]; then
        WHITE_FULL_URLS[$port]="${WHITE_URLS[$port]}$white_agent_url"
    else
        WHITE_FULL_URLS[$port]="${WHITE_URLS[$port]}/to_agent/<check_logs_$port>"
    fi
done

# ============================================
# 5. Update white_agents_config.json with Real Agent IDs
# ============================================
echo ""
echo "📝 Updating white_agents_config.json with real agent IDs from .ab directories..."

CONFIG_FILE="$PROJECT_ROOT/white_agents_config.json"

# Wait a bit for agent IDs to be written
sleep 3

# Create updated config with actual agent IDs from .ab directories
cat > "$CONFIG_FILE" << 'EOF'
{
  "description": "White agent URL to identifier mapping. Auto-updated by deployment script.",
  "agents": [
EOF

idx=0
for config in "${WHITE_CONFIGS[@]}"; do
    IFS='|' read -r dir port model use_rag <<< "$config"

    # Get Cloudflare base URL
    cf_base="${WHITE_URLS[$port]}"

    # Get actual agent ID from .ab directory
    agent_dir="$PROJECT_ROOT/$dir/.ab/agents"
    if [ -d "$agent_dir" ]; then
        # Get the subdirectory name (agent ID)
        agent_id=$(ls "$agent_dir" 2>/dev/null | head -1)
        if [ -n "$agent_id" ]; then
            # Construct full URL with real agent ID
            full_url="$cf_base/to_agent/$agent_id"
            echo "  ✓ $dir: Found agent ID $agent_id"
        else
            # Fallback: use placeholder
            full_url="$cf_base/to_agent/<agent_not_started>"
            echo "  ⚠️  $dir: No agent ID found yet"
        fi
    else
        # Fallback: use placeholder
        full_url="$cf_base/to_agent/<agent_not_started>"
        echo "  ⚠️  $dir: .ab directory not found"
    fi

    # Determine identifier based on model and mode
    model_clean=$(echo "$model" | sed 's/\//-/g' | sed 's/:/-/g' | sed 's/\./-/g')
    if [[ "$use_rag" == "true" ]]; then
        identifier="${model_clean}_rag"
    else
        identifier="${model_clean}_direct"
    fi

    # Determine mode string
    mode_str=$([[ "$use_rag" == "true" ]] && echo "rag" || echo "direct")

    # Add comma before entry if not first
    if [ $idx -gt 0 ]; then
        echo "," >> "$CONFIG_FILE"
    fi

    # Write agent entry (no trailing newline issues)
    printf '    {\n' >> "$CONFIG_FILE"
    printf '      "url": "%s",\n' "$full_url" >> "$CONFIG_FILE"
    printf '      "identifier": "%s",\n' "$identifier" >> "$CONFIG_FILE"
    printf '      "model": "%s",\n' "$model" >> "$CONFIG_FILE"
    printf '      "mode": "%s",\n' "$mode_str" >> "$CONFIG_FILE"
    printf '      "port": %d\n' "$port" >> "$CONFIG_FILE"
    printf '    }' >> "$CONFIG_FILE"

    idx=$((idx + 1))
done

# Close JSON (add newline before closing)
printf '\n  ]\n}\n' >> "$CONFIG_FILE"

echo "✅ Updated $CONFIG_FILE with real agent IDs"

# ============================================
# 6. Display Final Information
# ============================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Deployment Successful!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🟢 Green Agent (Evaluator):"
echo "   Public URL: $GREEN_URL"
echo "   Agent URL: $GREEN_FULL_URL"
echo "   Agent Card: ${GREEN_FULL_URL}/.well-known/agent-card.json"
echo "   Local Port: $GREEN_PORT"
echo ""

echo "⚪ White Agents:"
idx=1
for config in "${WHITE_CONFIGS[@]}"; do
    IFS='|' read -r dir port model use_rag <<< "$config"
    url="${WHITE_FULL_URLS[$port]}"
    mode=$([[ "$use_rag" == "true" ]] && echo "Web RAG" || echo "Direct LLM")

    echo ""
    echo "  White Agent #$idx - $dir - Port $port:"
    echo "   Model: $model"
    echo "   Mode: $mode"
    echo "   Public URL: ${WHITE_URLS[$port]}"
    echo "   Agent URL: $url"
    echo "   Agent Card: ${url}/.well-known/agent-card.json"

    idx=$((idx + 1))
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 Configuration Info (for AgentBeats Cloud Platform)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Green Agent:"
echo "  CLOUDRUN_HOST=$GREEN_DOMAIN"
echo "  HTTPS_ENABLED=true"
echo "  AGENT_TYPE=green"
echo ""

idx=1
for config in "${WHITE_CONFIGS[@]}"; do
    IFS='|' read -r dir port model use_rag <<< "$config"
    domain="${WHITE_DOMAINS[$port]}"
    use_direct=$([[ "$use_rag" == "true" ]] && echo "false" || echo "true")
    mode=$([[ "$use_rag" == "true" ]] && echo "Web RAG" || echo "Direct LLM")

    echo "White Agent #$idx ($dir - Port $port - $mode):"
    echo "  CLOUDRUN_HOST=$domain"
    echo "  HTTPS_ENABLED=true"
    echo "  AGENT_TYPE=white"
    echo "  WHITE_AGENT_MODEL=$model"
    echo "  USE_DIRECT_LLM=$use_direct"
    echo ""

    idx=$((idx + 1))
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Save URLs to file
{
    echo "Green Agent URL: $GREEN_FULL_URL"
    echo "Green Agent Card: ${GREEN_FULL_URL}/.well-known/agent-card.json"
    echo ""

    idx=1
    for config in "${WHITE_CONFIGS[@]}"; do
        IFS='|' read -r dir port model use_rag <<< "$config"
        url="${WHITE_FULL_URLS[$port]}"
        mode=$([[ "$use_rag" == "true" ]] && echo "Web RAG" || echo "Direct LLM")

        echo "White Agent #$idx ($dir - $model - $mode):"
        echo "  URL: $url"
        echo "  Card: ${url}/.well-known/agent-card.json"
        echo ""

        idx=$((idx + 1))
    done
} > /tmp/multi_agent_urls.txt

echo "💾 URLs saved to: /tmp/multi_agent_urls.txt"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 Ready to Evaluate!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ white_agents_config.json has been automatically updated"
echo "   with Cloudflare URLs!"
echo ""
echo "You can now send simple evaluation requests:"
echo ""
echo '  curl -X POST http://localhost:9001 \'
echo '    -H "Content-Type: application/json" \'
echo '    -d @test_evaluation_request.json'
echo ""
echo "Green agent will automatically look up model/mode for each URL."
echo ""
echo "📊 View logs:"
echo "  Green Agent: tail -f /tmp/green_agent.log"
for config in "${WHITE_CONFIGS[@]}"; do
    IFS='|' read -r dir port model use_rag <<< "$config"
    echo "  White Agent ($dir - Port $port): tail -f /tmp/white_agent_$port.log"
done
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Keep running
wait
