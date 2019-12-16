﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.RunTests;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests.Transformers
{
    public class CountTargetEncodingTests : TestDataPipeBase
    {
        public CountTargetEncodingTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void TestCountTargetEncodingEstimator()
        {
            string dataPath = GetDataPath("breast-cancer.txt");
            var data = ML.Data.LoadFromTextFile(dataPath, new[] {
                new TextLoader.Column("ScalarString", DataKind.String, 1),
                new TextLoader.Column("VectorString", DataKind.String, 1, 9),
                new TextLoader.Column("Label", DataKind.Single, 0)
            });

            var estimator = ML.Transforms.CountTargetEncode(new[] {
                new InputOutputColumnPair("ScalarString"), new InputOutputColumnPair("VectorString") }, "Label");
            TestEstimatorCore(estimator, data);
        }

        [Fact]
        public void TestSaveAndLoadExternalCounts()
        {
            var dataPath = GetDataPath("breast-cancer.txt");
            var data = ML.Data.LoadFromTextFile(dataPath, new[] { new TextLoader.Column("Label", DataKind.Single, 0),
                new TextLoader.Column("Text", DataKind.String, 1,9)});
            var estimator = ML.Transforms.CountTargetEncode("Text", builder: CountTableBuilderBase.CreateCMCountTableBuilder(2, 1 << 6));
            var transformer = estimator.Fit(data);

            estimator = ML.Transforms.CountTargetEncode("Text", transformer);
            var transformer1 = estimator.Fit(new EmptyDataView(Env, data.Schema));

            CheckSameCounts(data, transformer, transformer1, 3);
        }

        [Fact]
        public void TestSaveAndLoadExternalCountsMultipleColumns()
        {
            var dataPath = GetDataPath("breast-cancer.txt");
            var data = ML.Data.LoadFromTextFile(dataPath, new[] {
                new TextLoader.Column("ScalarString", DataKind.String, 1),
                new TextLoader.Column("VectorString", DataKind.String, 1, 9),
                new TextLoader.Column("Label", DataKind.Single, 0)
            });

            var estimator = ML.Transforms.CountTargetEncode(new[] {
                new InputOutputColumnPair("ScalarString"), new InputOutputColumnPair("VectorString") }, "Label", CountTableBuilderBase.CreateCMCountTableBuilder(2, 1 << 6));
            var transformer = estimator.Fit(data);

            estimator = ML.Transforms.CountTargetEncode(new[] { new InputOutputColumnPair("ScalarString"), new InputOutputColumnPair("VectorString") }, transformer);
            var transformer1 = estimator.Fit(new EmptyDataView(Env, data.Schema));

            CheckSameCounts(data, transformer, transformer1, 2, 5);
        }

        private static void CheckSameCounts(IDataView data, CountTargetEncodingTransformer transformer, CountTargetEncodingTransformer transformer1, params int[] countCols)
        {
            var transformedData = transformer.Transform(data);
            var transformedData1 = transformer1.Transform(data);

            using (var curs = transformedData.GetRowCursor(transformedData.Schema))
            using (var curs1 = transformedData1.GetRowCursor(transformedData.Schema))
            {
                var getters = new ValueGetter<VBuffer<float>>[countCols.Length];
                var getters1 = new ValueGetter<VBuffer<float>>[countCols.Length];
                for (int i = 0; i < countCols.Length; i++)
                {
                    getters[i] = curs.GetGetter<VBuffer<float>>(transformedData.Schema[countCols[i]]);
                    getters1[i] = curs1.GetGetter<VBuffer<float>>(transformedData1.Schema[countCols[i]]);
                }

                var buffer = default(VBuffer<float>);
                var buffer1 = default(VBuffer<float>);
                while (curs.MoveNext())
                {
                    Assert.True(curs1.MoveNext());
                    for (int i = 0; i < countCols.Length; i++)
                    {
                        getters[i](ref buffer);
                        getters1[i](ref buffer1);
                        var values = buffer.GetValues();
                        var values1 = buffer1.GetValues();
                        for (int label = 0; label < 2; label++)
                            Assert.Equal(values[label], values1[label]);
                    }
                }
            }
        }

        [Fact]
        public void TestCountTargetEncodingEstimatorWithBuilders()
        {
            var dataPath = GetDataPath("breast-cancer.txt");
            var data = ML.Data.LoadFromTextFile(dataPath, new[] {
                new TextLoader.Column("ScalarString", DataKind.String, 1),
                new TextLoader.Column("VectorString", DataKind.String, 1, 9),
                new TextLoader.Column("Label", DataKind.Single, 0)
            });

            var estimator = ML.Transforms.CountTargetEncode("VectorString1", "VectorString", builder: CountTableBuilderBase.CreateCMCountTableBuilder(3, 1 << 10),
                priorCoefficient: 0.5f, laplaceScale: 0.1f, numberOfBits: 25).Append(
                ML.Transforms.CountTargetEncode(new[] { new InputOutputColumnPair("ScalarString1", "ScalarString"), new InputOutputColumnPair("VectorString2", "VectorString") },
                "Label", sharedTable: true)).Append(
                ML.Transforms.CountTargetEncode("ScalarString2", "ScalarString", builder: CountTableBuilderBase.CreateDictionaryCountTableBuilder(1)));

            TestEstimatorCore(estimator, data);
        }
    }
}
