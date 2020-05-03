// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Numeric;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data.DataView
{
    public abstract class BatchTransformBase<TInput, TBatch> : IDataView
    {
        private protected sealed class Bindings : ColumnBindingsBase
        {
            private readonly DataViewType _outputColumnType;
            private readonly int _inputColumnIndex;

            public Bindings(DataViewSchema input, string inputColumnName, string outputColumnName, DataViewType outputColumnType)
                : base(input, true, outputColumnName)
            {
                _outputColumnType = outputColumnType;
                _inputColumnIndex = Input[inputColumnName].Index;
            }

            protected override DataViewType GetColumnTypeCore(int iinfo)
            {
                Contracts.Check(iinfo == 0);
                return _outputColumnType;
            }

            // Get a predicate for the input columns.
            public Func<int, bool> GetDependencies(Func<int, bool> predicate)
            {
                Contracts.AssertValue(predicate);

                var active = new bool[Input.Count];
                for (int col = 0; col < ColumnCount; col++)
                {
                    if (!predicate(col))
                        continue;

                    bool isSrc;
                    int index = MapColumnIndex(out isSrc, col);
                    if (isSrc)
                        active[index] = true;
                    else
                        active[_inputColumnIndex] = true;
                }

                return col => 0 <= col && col < active.Length && active[col];
            }
        }

        public bool CanShuffle => false;

        public DataViewSchema Schema => SchemaBindings.AsSchema;

        private readonly IDataView _source;
        private readonly IHost _host;
        private protected readonly Bindings SchemaBindings;
        protected readonly string InputCol;

        protected BatchTransformBase(IHostEnvironment env, IDataView input, string inputColumnName, string outputColumnName, DataViewType outputColumnType)
        {
            _host = env.Register("Batch");
            _source = input;
            SchemaBindings = new Bindings(input.Schema, inputColumnName, outputColumnName, outputColumnType);
            InputCol = inputColumnName;
        }

        public long? GetRowCount() => _source.GetRowCount();

        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            _host.CheckValue(columnsNeeded, nameof(columnsNeeded));
            _host.CheckValueOrNull(rand);

            var predicate = RowCursorUtils.FromColumnsToPredicate(columnsNeeded, SchemaBindings.AsSchema);

            // If we aren't selecting any of the output columns, don't construct our cursor.
            // Note that because we cannot support random due to the inherently
            // stratified nature, neither can we allow the base data to be shuffled,
            // even if it supports shuffling.
            if (!SchemaBindings.AnyNewColumnsActive(predicate))
            {
                var activeInput = SchemaBindings.GetActiveInput(predicate);
                var inputCursor = _source.GetRowCursor(_source.Schema.Where(c => activeInput[c.Index]), null);
                return new BindingsWrappedRowCursor(_host, inputCursor, SchemaBindings);
            }
            var active = SchemaBindings.GetActive(predicate);
            Contracts.Assert(active.Length == SchemaBindings.ColumnCount);

            // REVIEW: We can get a different input predicate for the input cursor and for the lookahead cursor. The lookahead
            // cursor is only used for getting the values from the input column, so it only needs that column activated. The
            // other cursor is used to get source columns, so it needs the rest of them activated.
            var predInput = SchemaBindings.GetDependencies(predicate);
            var inputCols = _source.Schema.Where(c => predInput(c.Index));
            return new Cursor(this, _source.GetRowCursor(inputCols), _source.GetRowCursor(inputCols), active);
        }

        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
        {
            return new[] { GetRowCursor(columnsNeeded, rand) };
        }

        protected abstract TBatch InitializeBatch(DataViewRowCursor input);
        protected abstract void ProcessBatch(TBatch currentBatch);
        protected abstract void ProcessExample(TBatch currentBatch, TInput currentInput);
        protected abstract Func<bool> GetLastInBatchDelegate(DataViewRowCursor lookAheadCursor);
        protected abstract Func<bool> GetIsNewBatchDelegate(DataViewRowCursor lookAheadCursor);
        protected abstract Delegate[] CreateGetters(DataViewRowCursor input, TBatch currentBatch, bool[] active);

        private sealed class Cursor : RootCursorBase
        {
            private readonly BatchTransformBase<TInput, TBatch> _parent;
            private readonly DataViewRowCursor _lookAheadCursor;
            private readonly DataViewRowCursor _input;

            private readonly bool[] _active;
            private readonly Delegate[] _getters;

            private TBatch _currentBatch;
            private readonly Func<bool> _lastInBatchInLookAheadCursorDel;
            private readonly Func<bool> _firstInBatchInInputCursorDel;
            private readonly ValueGetter<TInput> _inputGetterInLookAheadCursor;
            private TInput _currentInput;

            public override long Batch => 0;

            public override DataViewSchema Schema => _parent.Schema;

            public Cursor(BatchTransformBase<TInput, TBatch> parent, DataViewRowCursor input, DataViewRowCursor lookAheadCursor, bool[] active)
                : base(parent._host)
            {
                _parent = parent;
                _input = input;
                _lookAheadCursor = lookAheadCursor;
                _active = active;

                _currentBatch = _parent.InitializeBatch(_input);

                _getters = _parent.CreateGetters(_input, _currentBatch, _active);

                _lastInBatchInLookAheadCursorDel = _parent.GetLastInBatchDelegate(_lookAheadCursor);
                _firstInBatchInInputCursorDel = _parent.GetIsNewBatchDelegate(_input);
                _inputGetterInLookAheadCursor = _lookAheadCursor.GetGetter<TInput>(_lookAheadCursor.Schema[_parent.InputCol]);
            }

            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                Contracts.CheckParam(IsColumnActive(column), nameof(column), "requested column is not active");

                var col = _parent.SchemaBindings.MapColumnIndex(out bool isSrc, column.Index);
                if (isSrc)
                {
                    Contracts.AssertValue(_input);
                    return _input.GetGetter<TValue>(_input.Schema[col]);
                }

                Ch.AssertValue(_getters);
                var getter = _getters[col];
                Ch.Assert(getter != null);
                var fn = getter as ValueGetter<TValue>;
                if (fn == null)
                    throw Ch.Except("Invalid TValue in GetGetter: '{0}'", typeof(TValue));
                return fn;
            }

            public override ValueGetter<DataViewRowId> GetIdGetter()
            {
                return
                    (ref DataViewRowId val) =>
                    {
                        Ch.Check(IsGood, "Cannot call ID getter in current state");
                        val = new DataViewRowId((ulong)Position, 0);
                    };
            }

            public override bool IsColumnActive(DataViewSchema.Column column)
            {
                Ch.Check(column.Index < _parent.SchemaBindings.AsSchema.Count);
                return _active[column.Index];
            }

            protected override bool MoveNextCore()
            {
                if (!_input.MoveNext())
                    return false;
                if (!_firstInBatchInInputCursorDel())
                    return true;

                // If we are here, this means that _input.MoveNext() has gotten us to the beginning of the next batch,
                // so now we need to look ahead at the entire next batch in the _lookAheadCursor.
                // The _lookAheadCursor's position should be on the last row of the previous batch (or -1).
                Ch.Assert(_lastInBatchInLookAheadCursorDel());

                var good = _lookAheadCursor.MoveNext();
                // The two cursors should have the same number of elements, so if _input.MoveNext() returned true,
                // then it must return true here too.
                Ch.Assert(good);

                do
                {
                    _inputGetterInLookAheadCursor(ref _currentInput);
                    _parent.ProcessExample(_currentBatch, _currentInput);
                } while (!_lastInBatchInLookAheadCursorDel() && _lookAheadCursor.MoveNext());

                _parent.ProcessBatch(_currentBatch);
                return true;
            }
        }
    }

    // Sample class: look at the whole batch of float values <d0,d1,...,dn>, and return for each input x the
    // product x*dotproduct(<d0, x*d1,...,x*dn>).
    public sealed class BatchTransform : BatchTransformBase<float, BatchTransform.Batch>
    {
        private readonly int _batchSize;

        public BatchTransform(IHostEnvironment env, IDataView input, string inputColumnName, string outputColumnName, int batchSize)
            : base(env, input, inputColumnName, outputColumnName, new VectorDataViewType(NumberDataViewType.Double, batchSize))
        {
            _batchSize = batchSize;
        }

        protected override Delegate[] CreateGetters(DataViewRowCursor input, Batch currentBatch, bool[] active)
        {
            if (!SchemaBindings.AnyNewColumnsActive(x => active[x]))
                return new Delegate[1];
            return new[] { currentBatch.CreateGetter(input, InputCol) };
        }

        protected override Batch InitializeBatch(DataViewRowCursor input) => new Batch(_batchSize);

        protected override Func<bool> GetIsNewBatchDelegate(DataViewRowCursor input)
        {
            return () => input.Position % _batchSize == 0;
        }

        protected override Func<bool> GetLastInBatchDelegate(DataViewRowCursor input)
        {
            return () => (input.Position + 1) % _batchSize == 0;
        }

        protected override void ProcessExample(Batch currentBatch, float currentInput)
        {
            currentBatch.AddValue(currentInput);
        }

        protected override void ProcessBatch(Batch currentBatch)
        {
            // Here we would do any calculations that need to be done on the entire batch.
            currentBatch.Process();
            currentBatch.Reset();
        }

        public sealed class Batch
        {
            private readonly List<float> _batch;
            private float _dotProductCur;

            public Batch(int batchSize)
            {
                _batch = new List<float>(batchSize);
            }

            public void AddValue(float value)
            {
                _batch.Add(value);
            }

            public void Process()
            {
                _dotProductCur = VectorUtils.NormSquared(new ReadOnlySpan<float>(_batch.ToArray()));
            }

            public void Reset()
            {
                _batch.Clear();
            }

            public ValueGetter<float> CreateGetter(DataViewRowCursor input, string inputCol)
            {
                ValueGetter<float> srcGetter = input.GetGetter<float>(input.Schema[inputCol]);
                ValueGetter<float> getter =
                    (ref float dst) =>
                    {
                        float src = default;
                        srcGetter(ref src);
                        dst = src * _dotProductCur;
                    };
                return getter;
            }
        }
    }
}
