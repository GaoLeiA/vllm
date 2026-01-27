import time
from dataclasses import dataclass
from typing import List, Optional

# --- Mock Classes ---

@dataclass
class Request:
    request_id: str
    prompt_len: int
    output_target: int
    arrival_time: int
    
    # Internal state
    generated_len: int = 0
    status: str = "WAITING"  # WAITING, RUNNING, FINISHED

    def is_finished(self):
        return self.generated_len >= self.output_target

class MockScheduler:
    def __init__(self, max_batch_tokens: int):
        self.max_batch_tokens = max_batch_tokens
        self.waiting: List[Request] = []
        self.running: List[Request] = []
        self.current_time = 0

    def add_request(self, req: Request):
        print(f"[Time {self.current_time}] New Request Arrived: {req.request_id} (Prompt: {req.prompt_len})")
        self.waiting.append(req)

    def schedule(self):
        print(f"\n--- Step at Time {self.current_time} ---")
        
        # 1. Prioritize RUNNING requests (Decode phase)
        # In vLLM, we first allocate slots for requests that are already generating.
        batch = []
        token_budget = self.max_batch_tokens
        
        # Process running requests
        # (Using a copy or index to allow removal if finished inside loop, though usually done after)
        active_running = []
        for req in self.running:
            if req.is_finished():
                continue
            
            # Each decoding request consumes 1 slot per step
            if token_budget >= 1:
                batch.append(f"{req.request_id}(Decode)")
                req.generated_len += 1
                token_budget -= 1
                active_running.append(req)
            else:
                print(f"  [Scheduler] Preempted {req.request_id} due to budget!")
                self.waiting.insert(0, req) # Put back to front of waiting
                req.status = "WAITING"
        
        self.running = active_running

        # 2. Try to schedule WAITING requests (Prefill phase)
        # This is where "Continuous Batching" happens: inserting Prefills 
        # into the *remaining* budget of the current step.
        while self.waiting and token_budget > 0:
            req = self.waiting[0]
            
            # For a new request, we need to process the WHOLE prompt (simplified)
            # or chunks of it. Here we assume we must process at least 1 token if chunked,
            # or whole prompt if not. Let's assume simplified: need full prompt budget.
            needed = req.prompt_len
            
            if token_budget >= needed:
                self.waiting.pop(0)
                self.running.append(req)
                req.status = "RUNNING"
                batch.append(f"{req.request_id}(Prefill-{needed})")
                token_budget -= needed
            else:
                # Not enough space for this prefill
                print(f"  [Scheduler] Not enough budget for {req.request_id} prefill. Needed: {needed}, Left: {token_budget}")
                break
        
        # 3. Execute Batch
        if not batch:
            print("  [System] Idle.")
        else:
            print(f"  [GPU Executing Batch]: {batch}")
            print(f"  [Remaining Budget]: {token_budget}")

        # Check for completions
        finished_now = [r.request_id for r in self.running if r.is_finished()]
        if finished_now:
            print(f"  [Finished Requests]: {finished_now}")
            self.running = [r for r in self.running if not r.is_finished()]

    def tick(self):
        self.current_time += 1

# --- Simulation ---

def run_simulation():
    # Setup: Max 10 tokens processing at once
    scheduler = MockScheduler(max_batch_tokens=10)
    
    # Request definition: ID, PromptLen, OutputLen, ArrivalTime
    scenarios = [
        Request("ReqA", prompt_len=5, output_target=4, arrival_time=0),
        Request("ReqB", prompt_len=4, output_target=3, arrival_time=2), # Arrives while A is decoding
        Request("ReqC", prompt_len=8, output_target=2, arrival_time=3), # Big prefill
    ]
    
    # Simulation Loop
    max_steps = 10
    scenario_idx = 0
    
    for t in range(max_steps):
        # 1. Simulate new arrivals
        while scenario_idx < len(scenarios) and scenarios[scenario_idx].arrival_time == t:
            scheduler.add_request(scenarios[scenario_idx])
            scenario_idx += 1
            
        # 2. Schedule and Run
        scheduler.schedule()
        
        # 3. Advance time
        scheduler.tick()
        time.sleep(0.5) # Just for visuals if running interactively

if __name__ == "__main__":
    run_simulation()
