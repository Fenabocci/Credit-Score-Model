import { CreditScoreForm } from './components/CreditScoreForm'

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24">
      <h1 className="text-4xl font-bold mb-8">Credit Score Predictor</h1>
      <CreditScoreForm />
    </main>
  )
}

