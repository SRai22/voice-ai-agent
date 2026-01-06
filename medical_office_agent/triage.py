"""
---
title: Medical Office Triage System
category: complex-agents
tags: [multi_agent, agent_transfer, medical, context_preservation, chat_history]
difficulty: advanced
description: Multi-agent medical triage system with specialized departments
demonstrates:
  - Multiple specialized agents (triage, support, billing)
  - Agent-to-agent transfer with context preservation
  - Chat history truncation and management
  - Shared userdata across agent transfers
  - Room attribute updates for agent tracking
  - YAML prompt loading for agent instructions
---
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import RunContext
from livekit.plugins import openai, silero

from utils import load_prompt

logger = logging.getLogger("medical-office-triage")
logger.setLevel(logging.INFO)

load_dotenv()

# --- Server Setup ---
server = AgentServer(job_memory_warn_mb=1500)


def prewarm(proc: JobProcess):
    # Preload VAD
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm

@dataclass
class UserData:
    """Stores data and agents to be shared across the session"""
    personas: dict[str, Agent] = field(default_factory=dict)
    prev_agent: Optional[Agent] = None
    ctx: Optional[JobContext] = None
    transfer_history: list[str] = field(default_factory=list)
    max_transfer_history: int = 3

    def summarize(self) -> str:
        return "User data: Medical office triage system"

RunContext_T = RunContext[UserData]

class BaseAgent(Agent):
    async def on_enter(self) -> None:
        agent_name = self.__class__.__name__
        logger.info(f"Entering {agent_name}")

        userdata: UserData = self.session.userdata
        
        # Track this agent in transfer history
        userdata.transfer_history.append(agent_name)
        if len(userdata.transfer_history) > userdata.max_transfer_history:
            userdata.transfer_history.pop(0)
        
        logger.info(f"Transfer history: {' -> '.join(userdata.transfer_history)}")
        
        # Set room attributes only if connected
        if userdata.ctx and userdata.ctx.room and userdata.ctx.room.isconnected():
            try:
                await userdata.ctx.room.local_participant.set_attributes({"agent": agent_name})
            except Exception as e:
                logger.warning(f"Could not set agent attributes: {e}")

        chat_ctx = self.chat_ctx.copy()

        if userdata.prev_agent:
            items_copy = self._truncate_chat_ctx(
                userdata.prev_agent.chat_ctx.items, keep_function_call=True
            )
            existing_ids = {item.id for item in chat_ctx.items}
            items_copy = [item for item in items_copy if item.id not in existing_ids]
            chat_ctx.items.extend(items_copy)

        # Add the Persona System Prompt
        chat_ctx.add_message(
            role="system",
            content=f"You are the {agent_name}. {userdata.summarize()}"
        )

        # Add a specific "Handoff Trigger" if this is a transfer
        if userdata.prev_agent:
            prev_agent_name = userdata.prev_agent.__class__.__name__
            recent_agents = ', '.join(userdata.transfer_history[-3:])
            chat_ctx.add_message(
                role="system",
                content=f"CRITICAL: The user was just transferred to you ({agent_name}) from {prev_agent_name}. Recent path: {recent_agents}. You MUST handle their request yourself. NEVER transfer back to {prev_agent_name}. Only transfer if the user explicitly requests a completely different department that you cannot help with. If their question is remotely related to your role, answer it directly."
            )
            
        await self.update_chat_ctx(chat_ctx)
        self.session.generate_reply()

    def _truncate_chat_ctx(
        self,
        items: list,
        keep_last_n_messages: int = 6,
        keep_system_message: bool = False,
        keep_function_call: bool = False,
    ) -> list:
        """Truncate the chat context to keep the last n messages."""
        def _valid_item(item) -> bool:
            if not keep_system_message and item.type == "message" and item.role == "system":
                return False
            if not keep_function_call and item.type in ["function_call", "function_call_output"]:
                return False
            return True

        new_items = []
        for item in reversed(items):
            if _valid_item(item):
                new_items.append(item)
            if len(new_items) >= keep_last_n_messages:
                break
        new_items = new_items[::-1]

        while new_items and new_items[0].type in ["function_call", "function_call_output"]:
            new_items.pop(0)

        return new_items

    async def _transfer_to_agent(self, name: str, context: RunContext_T) -> Agent:
        """Transfer to another agent while preserving context"""
        userdata = context.userdata
        current_agent = context.session.current_agent
        next_agent = userdata.personas[name]
        
        # Check for transfer loops
        target_agent_name = next_agent.__class__.__name__
        if len(userdata.transfer_history) >= 2:
            prev_agent_name = userdata.transfer_history[-1]
            if target_agent_name == prev_agent_name:
                logger.warning(f"Preventing immediate transfer back to {target_agent_name}")
                await self.session.say("I apologize for the confusion. Let me help you directly with your question instead.")
                return current_agent
        
        # Check for A->B->A pattern
        if len(userdata.transfer_history) >= 2:
            if target_agent_name == userdata.transfer_history[-2]:
                logger.warning(f"Preventing transfer loop: {userdata.transfer_history[-2]} -> {userdata.transfer_history[-1]} -> {target_agent_name}")
                await self.session.say("I see you've been transferred between departments. Let me assist you directly with your question.")
                return current_agent
        
        userdata.prev_agent = current_agent
        logger.info(f"Transferring from {current_agent.__class__.__name__} to {target_agent_name}")

        return next_agent


class TriageAgent(BaseAgent):
    def __init__(self, vad_inst) -> None:
        super().__init__(
            instructions=load_prompt('triage_prompt.yaml'),
            stt=openai.STT(
                base_url=os.getenv("WHISPER_URL", "http://localhost:11435/v1"),
                model="Systran/faster-whisper-small",
            ),
            llm=openai.LLM(
                base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
                model="qwen2.5:3b",
                timeout=30,
                temperature=0.1
            ),
            tts=openai.TTS(
                base_url=os.getenv("KOKORO_URL", "http://localhost:8880/v1"),
                model="tts-1",
                voice="af_sarah",
                speed=1.0,
            ), 
            vad=vad_inst
        )

    @function_tool
    async def transfer_to_support(self, context: RunContext_T) -> Agent:
        await self.session.say("I'll transfer you to our Patient Support team who can help with your medical services request.")
        return await self._transfer_to_agent("support", context)

    @function_tool
    async def transfer_to_billing(self, context: RunContext_T) -> Agent:
        await self.session.say("I'll transfer you to our Medical Billing department who can assist with your insurance and payment questions.")
        return await self._transfer_to_agent("billing", context)


class SupportAgent(BaseAgent):
    def __init__(self, vad_inst) -> None:
        super().__init__(
            instructions=load_prompt('support_prompt.yaml'),
            stt=openai.STT(
                base_url=os.getenv("WHISPER_URL", "http://localhost:11435/v1"),
                model="Systran/faster-whisper-small",
            ),
            llm=openai.LLM(
                base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
                model="qwen2.5:3b",
                timeout=30,
                temperature=0.05
            ),
            tts=openai.TTS(
                base_url=os.getenv("KOKORO_URL", "http://localhost:8880/v1"),
                model="tts-1",
                voice="am_adam",
                speed=1.0,
            ), 
            vad=vad_inst
        )

    @function_tool
    async def transfer_to_triage(self, context: RunContext_T) -> Agent:
        await self.session.say("I'll transfer you back to our Medical Office Triage agent who can better direct your inquiry.")
        return await self._transfer_to_agent("triage", context)

    @function_tool
    async def transfer_to_billing(self, context: RunContext_T) -> Agent:
        await self.session.say("I'll transfer you to our Medical Billing department for assistance with your insurance and payment questions.")
        return await self._transfer_to_agent("billing", context)


class BillingAgent(BaseAgent):
    def __init__(self, vad_inst) -> None:
        super().__init__(
            instructions=load_prompt('billing_prompt.yaml'),
            stt=openai.STT(
                base_url=os.getenv("WHISPER_URL", "http://localhost:11435/v1"),
                model="Systran/faster-whisper-small",
            ),
            llm=openai.LLM(
                base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
                model="qwen2.5:3b",
                timeout=30,
                temperature=0.05
            ),
            tts=openai.TTS(
                base_url=os.getenv("KOKORO_URL", "http://localhost:8880/v1"),
                model="tts-1",
                voice="af_nova",
                speed=1.0,
            ), 
            vad=vad_inst
        )

    @function_tool
    async def transfer_to_triage(self, context: RunContext_T) -> Agent:
        await self.session.say("I'll transfer you back to our Medical Office Triage agent who can better direct your inquiry.")
        return await self._transfer_to_agent("triage", context)

    @function_tool
    async def transfer_to_support(self, context: RunContext_T) -> Agent:
        await self.session.say("I'll transfer you to our Patient Support team who can help with your medical services request.")
        return await self._transfer_to_agent("support", context)


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # Retrieve prewarmed VAD
    vad_inst = ctx.proc.userdata["vad"]
    
    userdata = UserData(ctx=ctx)
    triage_agent = TriageAgent(vad_inst)
    support_agent = SupportAgent(vad_inst)
    billing_agent = BillingAgent(vad_inst)

    # Register all agents in the userdata
    userdata.personas.update({
        "triage": triage_agent,
        "support": support_agent,
        "billing": billing_agent
    })

    session = AgentSession[UserData](userdata=userdata)

    await session.start(
        agent=triage_agent,  # Start with the Medical Office Triage agent
        room=ctx.room,
    )

if __name__ == "__main__":
    cli.run_app(server)