import { useParams } from 'react-router-dom';

export default function TraceDetail() {
  const { traceId } = useParams();

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
        Trace Detail
      </h1>
      <p className="mt-2 font-mono text-sm text-gray-500">
        {traceId}
      </p>

      <div className="mt-8 card p-6">
        <p className="text-gray-500">
          Trace detail view coming soon...
        </p>
      </div>
    </div>
  );
}
