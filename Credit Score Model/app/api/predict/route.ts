import { NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export async function POST(req: Request) {
  try {
    const body = await req.json();
    
    // Convert the JSON body to a string of comma-separated values
    const inputData = Object.values(body).join(',');
    
    // Run the Python script with the input data
    const { stdout, stderr } = await execAsync(`python credit_score_model.py ${inputData}`);
    
    if (stderr) {
      console.error('Error:', stderr);
      return NextResponse.json({ error: 'An error occurred while processing your request.' }, { status: 500 });
    }
    
    // Parse the output from the Python script
    const prediction = JSON.parse(stdout);
    
    return NextResponse.json(prediction);
  } catch (error) {
    console.error('Error:', error);
    return NextResponse.json({ error: 'An error occurred while processing your request.' }, { status: 500 });
  }
}

