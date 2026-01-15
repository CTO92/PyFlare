/**
 * Budget Configuration Component
 * Form for creating and editing cost budgets
 */

import { useState } from 'react';
import { Plus, Trash2, Save, AlertCircle } from 'lucide-react';
import { clsx } from 'clsx';
import type { Budget } from './BudgetTracker';

interface BudgetConfigProps {
  budgets: Budget[];
  onSave: (budgets: Budget[]) => Promise<void>;
  onDelete: (budgetId: string) => Promise<void>;
  availableModels?: string[];
  availableServices?: string[];
  availableUsers?: string[];
}

type BudgetFormData = Omit<Budget, 'id' | 'used' | 'resetDate'>;

const DEFAULT_BUDGET: BudgetFormData = {
  name: '',
  limit: 1000,
  period: 'monthly',
  alertThreshold: 80,
  scope: { type: 'global' },
};

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

function BudgetForm({
  budget,
  onChange,
  onRemove,
  isNew,
  availableModels = [],
  availableServices = [],
  availableUsers = [],
}: {
  budget: BudgetFormData;
  onChange: (budget: BudgetFormData) => void;
  onRemove?: () => void;
  isNew?: boolean;
  availableModels?: string[];
  availableServices?: string[];
  availableUsers?: string[];
}) {
  const scopeOptions = budget.scope?.type === 'model'
    ? availableModels
    : budget.scope?.type === 'service'
      ? availableServices
      : budget.scope?.type === 'user'
        ? availableUsers
        : [];

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-gray-900">
      <div className="flex items-start justify-between">
        <h4 className="font-medium text-gray-900 dark:text-white">
          {isNew ? 'New Budget' : budget.name || 'Unnamed Budget'}
        </h4>
        {onRemove && (
          <button
            onClick={onRemove}
            className="text-gray-400 hover:text-red-500"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        )}
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
        {/* Budget Name */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            Budget Name
          </label>
          <input
            type="text"
            value={budget.name}
            onChange={(e) => onChange({ ...budget, name: e.target.value })}
            className="input mt-1 w-full"
            placeholder="e.g., Production API Budget"
          />
        </div>

        {/* Limit */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            Budget Limit (USD)
          </label>
          <input
            type="number"
            value={budget.limit}
            onChange={(e) => onChange({ ...budget, limit: Number(e.target.value) })}
            className="input mt-1 w-full"
            min={0}
            step={100}
          />
        </div>

        {/* Period */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            Budget Period
          </label>
          <select
            value={budget.period}
            onChange={(e) =>
              onChange({ ...budget, period: e.target.value as Budget['period'] })
            }
            className="input mt-1 w-full"
          >
            <option value="daily">Daily</option>
            <option value="weekly">Weekly</option>
            <option value="monthly">Monthly</option>
          </select>
        </div>

        {/* Alert Threshold */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            Alert Threshold (%)
          </label>
          <input
            type="number"
            value={budget.alertThreshold}
            onChange={(e) =>
              onChange({ ...budget, alertThreshold: Number(e.target.value) })
            }
            className="input mt-1 w-full"
            min={0}
            max={100}
            step={5}
          />
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            Alert when usage exceeds this percentage
          </p>
        </div>

        {/* Scope Type */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            Budget Scope
          </label>
          <select
            value={budget.scope?.type || 'global'}
            onChange={(e) =>
              onChange({
                ...budget,
                scope: { type: e.target.value as Budget['scope']['type'] },
              })
            }
            className="input mt-1 w-full"
          >
            <option value="global">Global (All Usage)</option>
            <option value="model">Per Model</option>
            <option value="service">Per Service</option>
            <option value="user">Per User</option>
          </select>
        </div>

        {/* Scope Value */}
        {budget.scope?.type !== 'global' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              {budget.scope?.type === 'model'
                ? 'Model'
                : budget.scope?.type === 'service'
                  ? 'Service'
                  : 'User'}
            </label>
            {scopeOptions.length > 0 ? (
              <select
                value={budget.scope?.value || ''}
                onChange={(e) =>
                  onChange({
                    ...budget,
                    scope: { ...budget.scope!, value: e.target.value },
                  })
                }
                className="input mt-1 w-full"
              >
                <option value="">Select...</option>
                {scopeOptions.map((opt) => (
                  <option key={opt} value={opt}>
                    {opt}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="text"
                value={budget.scope?.value || ''}
                onChange={(e) =>
                  onChange({
                    ...budget,
                    scope: { ...budget.scope!, value: e.target.value },
                  })
                }
                className="input mt-1 w-full"
                placeholder={`Enter ${budget.scope?.type} name`}
              />
            )}
          </div>
        )}
      </div>

      {/* Preview */}
      <div className="mt-4 rounded-lg bg-gray-50 p-3 dark:bg-gray-800">
        <p className="text-sm text-gray-600 dark:text-gray-400">
          Budget: {formatCurrency(budget.limit)} per {budget.period}
          {budget.scope?.type !== 'global' &&
            budget.scope?.value &&
            ` for ${budget.scope.type}: ${budget.scope.value}`}
          . Alert at {budget.alertThreshold}% usage.
        </p>
      </div>
    </div>
  );
}

export default function BudgetConfig({
  budgets: initialBudgets,
  onSave,
  onDelete,
  availableModels = [],
  availableServices = [],
  availableUsers = [],
}: BudgetConfigProps) {
  const [budgets, setBudgets] = useState<(Budget | BudgetFormData)[]>(
    initialBudgets.length > 0 ? initialBudgets : []
  );
  const [newBudgets, setNewBudgets] = useState<BudgetFormData[]>([]);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const handleAddBudget = () => {
    setNewBudgets([...newBudgets, { ...DEFAULT_BUDGET }]);
  };

  const handleUpdateExisting = (index: number, updated: Budget | BudgetFormData) => {
    const newList = [...budgets];
    newList[index] = updated;
    setBudgets(newList);
  };

  const handleUpdateNew = (index: number, updated: BudgetFormData) => {
    const newList = [...newBudgets];
    newList[index] = updated;
    setNewBudgets(newList);
  };

  const handleRemoveNew = (index: number) => {
    setNewBudgets(newBudgets.filter((_, i) => i !== index));
  };

  const handleDeleteExisting = async (budgetId: string) => {
    setDeletingId(budgetId);
    setError(null);
    try {
      await onDelete(budgetId);
      setBudgets(budgets.filter((b) => 'id' in b && b.id !== budgetId));
    } catch {
      setError('Failed to delete budget');
    } finally {
      setDeletingId(null);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);

    // Validate
    const allBudgets = [...budgets, ...newBudgets];
    for (const budget of allBudgets) {
      if (!budget.name.trim()) {
        setError('All budgets must have a name');
        setSaving(false);
        return;
      }
      if (budget.limit <= 0) {
        setError('Budget limit must be greater than 0');
        setSaving(false);
        return;
      }
    }

    try {
      // Add IDs to new budgets
      const budgetsToSave = allBudgets.map((b) => {
        if ('id' in b) return b;
        return {
          ...b,
          id: `budget-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
          used: 0,
          resetDate: new Date().toISOString(),
        } as Budget;
      });

      await onSave(budgetsToSave);
      setBudgets(budgetsToSave);
      setNewBudgets([]);
    } catch {
      setError('Failed to save budgets');
    } finally {
      setSaving(false);
    }
  };

  const hasChanges = newBudgets.length > 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="font-medium text-gray-900 dark:text-white">
            Budget Configuration
          </h3>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Set up cost budgets to track and control spending
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleAddBudget}
            className="btn-secondary flex items-center gap-2"
          >
            <Plus className="h-4 w-4" />
            Add Budget
          </button>
          <button
            onClick={handleSave}
            disabled={saving || !hasChanges}
            className={clsx(
              'btn-primary flex items-center gap-2',
              (saving || !hasChanges) && 'opacity-50'
            )}
          >
            <Save className="h-4 w-4" />
            {saving ? 'Saving...' : 'Save Changes'}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 rounded-lg bg-red-50 p-4 text-red-700 dark:bg-red-900/20 dark:text-red-400">
          <AlertCircle className="h-5 w-5" />
          {error}
        </div>
      )}

      {/* Existing Budgets */}
      {budgets.length > 0 && (
        <div className="space-y-4">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Existing Budgets
          </h4>
          {budgets.map((budget, index) => (
            <div key={'id' in budget ? budget.id : index} className="relative">
              <BudgetForm
                budget={budget}
                onChange={(updated) => handleUpdateExisting(index, updated)}
                onRemove={
                  'id' in budget
                    ? () => handleDeleteExisting(budget.id)
                    : undefined
                }
                availableModels={availableModels}
                availableServices={availableServices}
                availableUsers={availableUsers}
              />
              {deletingId === ('id' in budget ? budget.id : null) && (
                <div className="absolute inset-0 flex items-center justify-center rounded-lg bg-white/80 dark:bg-gray-900/80">
                  <span className="text-sm text-gray-500">Deleting...</span>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* New Budgets */}
      {newBudgets.length > 0 && (
        <div className="space-y-4">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
            New Budgets
          </h4>
          {newBudgets.map((budget, index) => (
            <BudgetForm
              key={index}
              budget={budget}
              onChange={(updated) => handleUpdateNew(index, updated)}
              onRemove={() => handleRemoveNew(index)}
              isNew
              availableModels={availableModels}
              availableServices={availableServices}
              availableUsers={availableUsers}
            />
          ))}
        </div>
      )}

      {/* Empty State */}
      {budgets.length === 0 && newBudgets.length === 0 && (
        <div className="rounded-lg border-2 border-dashed border-gray-200 p-8 text-center dark:border-gray-700">
          <p className="text-gray-500 dark:text-gray-400">
            No budgets configured. Click "Add Budget" to create one.
          </p>
        </div>
      )}
    </div>
  );
}
