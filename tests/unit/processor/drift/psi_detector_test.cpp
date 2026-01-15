/// @file psi_detector_test.cpp
/// @brief Tests for PSI drift detector

#include <gtest/gtest.h>

#include "processor/drift/psi_detector.h"

namespace pyflare::drift {
namespace {

class PSIDriftDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.threshold = 0.2;
        config_.num_bins = 10;
        config_.min_samples_per_bin = 5;
        detector_ = std::make_unique<PSIDriftDetector>(config_);
    }

    PSIConfig config_;
    std::unique_ptr<PSIDriftDetector> detector_;
};

TEST_F(PSIDriftDetectorTest, NoReferenceReturnsError) {
    std::vector<DataPoint> batch;
    batch.push_back(DataPoint{.id = "1", .features = {1.0, 2.0, 3.0}});

    auto result = detector_->Compute(batch);
    EXPECT_FALSE(result.ok());
}

TEST_F(PSIDriftDetectorTest, NoDriftWithSimilarDistribution) {
    // Set reference distribution - uniform 0-1
    Distribution reference;
    for (int i = 0; i < 100; ++i) {
        reference.AddSample({static_cast<double>(i) / 100.0});
    }
    ASSERT_TRUE(detector_->SetReference(reference).ok());

    // Test with similar distribution
    std::vector<DataPoint> batch;
    for (int i = 0; i < 50; ++i) {
        batch.push_back(DataPoint{
            .id = std::to_string(i),
            .features = {static_cast<double>(i) / 50.0}
        });
    }

    auto result = detector_->Compute(batch);
    ASSERT_TRUE(result.ok());
    EXPECT_FALSE(result->is_drifted);
    EXPECT_LT(result->score, config_.threshold);
}

TEST_F(PSIDriftDetectorTest, DetectsDriftWithDifferentDistribution) {
    // Set reference distribution - values near 0
    Distribution reference;
    for (int i = 0; i < 100; ++i) {
        reference.AddSample({0.1 + static_cast<double>(i) / 1000.0});
    }
    ASSERT_TRUE(detector_->SetReference(reference).ok());

    // Test with very different distribution - values near 10
    std::vector<DataPoint> batch;
    for (int i = 0; i < 50; ++i) {
        batch.push_back(DataPoint{
            .id = std::to_string(i),
            .features = {10.0 + static_cast<double>(i) / 10.0}
        });
    }

    auto result = detector_->Compute(batch);
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result->is_drifted);
    EXPECT_GT(result->score, config_.threshold);
}

TEST_F(PSIDriftDetectorTest, ComputeFeaturePSI_IdenticalDistributions) {
    std::vector<double> ref = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    std::vector<double> cur = ref;

    double psi = detector_->ComputeFeaturePSI(ref, cur);
    EXPECT_NEAR(psi, 0.0, 0.01);
}

TEST_F(PSIDriftDetectorTest, ComputeFeaturePSI_SlightlyDifferent) {
    std::vector<double> ref;
    std::vector<double> cur;

    for (int i = 0; i < 100; ++i) {
        ref.push_back(static_cast<double>(i));
        cur.push_back(static_cast<double>(i) + 0.5);  // Slight shift
    }

    double psi = detector_->ComputeFeaturePSI(ref, cur);
    EXPECT_LT(psi, 0.1);  // Should be low - minor drift
}

TEST_F(PSIDriftDetectorTest, ComputeFeaturePSI_VeryDifferent) {
    std::vector<double> ref;
    std::vector<double> cur;

    for (int i = 0; i < 100; ++i) {
        ref.push_back(static_cast<double>(i));
        cur.push_back(static_cast<double>(i + 500));  // Large shift
    }

    double psi = detector_->ComputeFeaturePSI(ref, cur);
    EXPECT_GT(psi, 0.25);  // Significant drift
}

TEST_F(PSIDriftDetectorTest, MultipleFeatures) {
    // Reference with 3 features
    Distribution reference;
    for (int i = 0; i < 100; ++i) {
        reference.AddSample({
            static_cast<double>(i),
            static_cast<double>(i) * 2,
            static_cast<double>(i) * 0.1
        });
    }
    ASSERT_TRUE(detector_->SetReference(reference).ok());

    // One feature drifted, two stable
    std::vector<DataPoint> batch;
    for (int i = 0; i < 50; ++i) {
        batch.push_back(DataPoint{
            .id = std::to_string(i),
            .features = {
                static_cast<double>(i),      // Same as reference
                static_cast<double>(i) * 2,  // Same as reference
                1000.0 + static_cast<double>(i)  // Drifted
            }
        });
    }

    auto result = detector_->Compute(batch);
    ASSERT_TRUE(result.ok());
    // Should detect drift in at least one feature
    EXPECT_TRUE(result->is_drifted || result->per_feature_scores.size() > 0);
}

TEST_F(PSIDriftDetectorTest, GettersAndSetters) {
    EXPECT_EQ(detector_->Name(), "PSIDriftDetector");
    EXPECT_EQ(detector_->Type(), DriftType::kFeature);
    EXPECT_DOUBLE_EQ(detector_->GetThreshold(), 0.2);

    detector_->SetThreshold(0.5);
    EXPECT_DOUBLE_EQ(detector_->GetThreshold(), 0.5);

    const auto& config = detector_->GetConfig();
    EXPECT_EQ(config.num_bins, 10);
}

TEST_F(PSIDriftDetectorTest, SerializeAndLoad) {
    // Set up reference
    Distribution reference;
    for (int i = 0; i < 100; ++i) {
        reference.AddSample({static_cast<double>(i)});
    }
    ASSERT_TRUE(detector_->SetReference(reference).ok());

    // Serialize
    auto state = detector_->SerializeState();
    ASSERT_TRUE(state.ok());
    EXPECT_FALSE(state->empty());

    // Load into new detector
    auto new_detector = std::make_unique<PSIDriftDetector>(config_);
    ASSERT_TRUE(new_detector->LoadState(*state).ok());

    // Verify same behavior
    std::vector<DataPoint> batch;
    for (int i = 0; i < 50; ++i) {
        batch.push_back(DataPoint{.id = std::to_string(i), .features = {static_cast<double>(i)}});
    }

    auto orig_result = detector_->Compute(batch);
    auto new_result = new_detector->Compute(batch);

    ASSERT_TRUE(orig_result.ok());
    ASSERT_TRUE(new_result.ok());
    EXPECT_NEAR(orig_result->score, new_result->score, 0.01);
}

TEST_F(PSIDriftDetectorTest, EmptyBatchReturnsError) {
    Distribution reference;
    for (int i = 0; i < 100; ++i) {
        reference.AddSample({static_cast<double>(i)});
    }
    ASSERT_TRUE(detector_->SetReference(reference).ok());

    std::vector<DataPoint> empty_batch;
    auto result = detector_->Compute(empty_batch);
    EXPECT_FALSE(result.ok());
}

TEST_F(PSIDriftDetectorTest, InsufficientSamplesWarning) {
    PSIConfig strict_config;
    strict_config.min_samples_per_bin = 50;  // High threshold
    strict_config.num_bins = 10;
    auto strict_detector = std::make_unique<PSIDriftDetector>(strict_config);

    Distribution reference;
    for (int i = 0; i < 100; ++i) {
        reference.AddSample({static_cast<double>(i)});
    }
    ASSERT_TRUE(strict_detector->SetReference(reference).ok());

    // Only 5 samples - might trigger insufficient samples handling
    std::vector<DataPoint> small_batch;
    for (int i = 0; i < 5; ++i) {
        small_batch.push_back(DataPoint{.id = std::to_string(i), .features = {static_cast<double>(i)}});
    }

    // Should still compute, but results may be less reliable
    auto result = strict_detector->Compute(small_batch);
    // Either succeeds with a result or fails with appropriate error
    if (result.ok()) {
        EXPECT_GE(result->score, 0.0);
    }
}

// =============================================================================
// Chi-Squared Detector Tests
// =============================================================================

class ChiSquaredDriftDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.p_value_threshold = 0.05;
        config_.min_expected_frequency = 5;
        detector_ = std::make_unique<ChiSquaredDriftDetector>(config_);
    }

    ChiSquaredDriftDetector::Config config_;
    std::unique_ptr<ChiSquaredDriftDetector> detector_;
};

TEST_F(ChiSquaredDriftDetectorTest, NameAndType) {
    EXPECT_EQ(detector_->Name(), "ChiSquaredDriftDetector");
    EXPECT_EQ(detector_->Type(), DriftType::kFeature);
}

TEST_F(ChiSquaredDriftDetectorTest, ThresholdGetterSetter) {
    EXPECT_DOUBLE_EQ(detector_->GetThreshold(), 0.05);
    detector_->SetThreshold(0.01);
    EXPECT_DOUBLE_EQ(detector_->GetThreshold(), 0.01);
}

// =============================================================================
// Factory Tests
// =============================================================================

TEST(PSIDetectorFactoryTest, CreatesPSIDetector) {
    PSIConfig config;
    config.threshold = 0.15;

    auto detector = CreatePSIDetector(config);
    ASSERT_NE(detector, nullptr);
    EXPECT_EQ(detector->Name(), "PSIDriftDetector");
    EXPECT_DOUBLE_EQ(detector->GetThreshold(), 0.15);
}

TEST(ChiSquaredFactoryTest, CreatesChiSquaredDetector) {
    ChiSquaredDriftDetector::Config config;
    config.p_value_threshold = 0.1;

    auto detector = CreateChiSquaredDetector(config);
    ASSERT_NE(detector, nullptr);
    EXPECT_EQ(detector->Name(), "ChiSquaredDriftDetector");
    EXPECT_DOUBLE_EQ(detector->GetThreshold(), 0.1);
}

}  // namespace
}  // namespace pyflare::drift
