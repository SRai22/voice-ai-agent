import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    // Get the request body (SDP offer from client)
    const body = await request.json();

    // Forward the offer to the agent backend
    const agentUrl = process.env.AGENT_URL || 'http://localhost:7860/api/offer';
    
    const response = await fetch(agentUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    // Check if the request was successful
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`Agent API error: ${response.status} - ${errorText}`);
      throw new Error(`Agent API responded with status: ${response.status}`);
    }

    // Get the JSON response (SDP answer from agent)
    const data = await response.json();

    // Return the answer to the client
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error in offer API route:', error);
    return NextResponse.json(
      { error: 'Failed to connect to voice agent', details: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}

export async function PATCH(request: NextRequest) {
  try {
    // Get the request body (ICE candidates or renegotiation from client)
    const body = await request.json();

    // Forward the PATCH request to the agent backend
    const agentUrl = process.env.AGENT_URL || 'http://localhost:7860/api/offer';
    
    const response = await fetch(agentUrl, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    // Check if the request was successful
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`Agent API PATCH error: ${response.status} - ${errorText}`);
      throw new Error(`Agent API PATCH responded with status: ${response.status}`);
    }

    // Get the JSON response
    const data = await response.json();

    // Return the response to the client
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error in offer PATCH API route:', error);
    return NextResponse.json(
      { error: 'Failed to update connection', details: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}
